"""Flask application serving a simple cycling power/pacing calculator."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import gpxpy
import gpxpy.gpx
import numpy as np
from flask import Flask, Response, flash, redirect, render_template, request, url_for

import firebase_admin
from firebase_admin import credentials, firestore

# --- Flask setup ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "dev-secret"  # In production use a proper secret key

# --- Firebase setup ------------------------------------------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Constants -----------------------------------------------------------------
SEGMENT_LENGTH_METERS = 200.0
GRAVITY = 9.81
RHO_AIR = 1.225
CDA = 0.30
CRR = 0.004
MIN_SPEED = 0.5  # m/s
MAX_SPEED = 25.0  # m/s (~90 km/h)
SMOOTHING_WINDOW = 7  # odd number for moving average
GRADE_CLAMP = 20.0
RPE_TO_PERCENT = {
    1: 0.35, 2: 0.50, 3: 0.60, 4: 0.70, 5: 0.80,
    6: 0.90, 7: 1.00, 8: 1.05, 9: 1.15, 10: 1.30,
}

LAST_CSV_ROWS: List[Dict[str, str]] = []

# --- Dataclasses ---------------------------------------------------------------
@dataclass
class GPXPoint:
    lat: float
    lon: float
    ele: float
    distance_m: float

@dataclass
class TrackLeg:
    start_distance: float
    end_distance: float
    delta_distance: float
    grade_percent: float
    delta_elevation: float

@dataclass
class SegmentResult:
    index: int
    km_from: str
    km_to: str
    length_m: str
    grade_percent: str
    power_w: str
    speed_kmh: str
    time_str: str

@dataclass
class SummaryResult:
    total_distance_km: str
    total_ascent: str
    total_descent: str
    avg_speed_kmh: str
    total_time: str

@dataclass
class ChartData:
    distance_km: List[float]
    power_w: List[float]
    grade_percent: List[float]

@dataclass
class CalculationResult:
    segments: List[SegmentResult]
    summary: SummaryResult
    chart: ChartData


# --- Utility functions ---------------------------------------------------------
def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

def moving_average(data: Iterable[float], window: int) -> np.ndarray:
    arr = np.array(list(data), dtype=float)
    if len(arr) == 0: return arr
    window = window if window % 2 == 1 else window + 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")

def parse_gpx(gpx_content: str) -> List[GPXPoint]:
    try:
        gpx = gpxpy.parse(gpx_content)
    except Exception as exc:
        raise ValueError("Soubor se nepodařilo načíst jako GPX.") from exc

    raw_points = [(p.latitude, p.longitude, p.elevation)
                  for t in gpx.tracks for s in t.segments for p in s.points if p.elevation]
    if not raw_points:
        raise ValueError("Trasa neobsahuje nadmořskou výšku nebo body.")

    filtered, last_lat, last_lon = [], None, None
    for lat, lon, ele in raw_points:
        if last_lat is None:
            filtered.append((lat, lon, ele))
        else:
            distance = gpxpy.geo.haversine_distance(last_lat, last_lon, lat, lon)
            if distance and distance >= 0.5:
                filtered.append((lat, lon, ele))
        last_lat, last_lon = lat, lon

    if len(filtered) < 2:
        raise ValueError("Trasa je příliš krátká pro segmentaci.")

    lats, lons, elevations = zip(*filtered)
    distances, total = [0.0], 0.0
    for i in range(1, len(filtered)):
        dist = gpxpy.geo.haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i]) or 0.0
        total += dist
        distances.append(total)
    if total < SEGMENT_LENGTH_METERS:
        raise ValueError("Trasa je příliš krátká pro segmentaci.")

    smoothed = moving_average(elevations, SMOOTHING_WINDOW)
    return [GPXPoint(lat, lon, ele, dist) for lat, lon, ele, dist in zip(lats, lons, smoothed, distances)]

def build_legs(points: List[GPXPoint]) -> Tuple[List[TrackLeg], float, float]:
    legs, asc, desc = [], 0.0, 0.0
    for prev, curr in zip(points, points[1:]):
        d = curr.distance_m - prev.distance_m
        if d <= 0.5: continue
        delta_e = curr.ele - prev.ele
        asc += max(delta_e, 0)
        desc += max(-delta_e, 0)
        grade = max(-GRADE_CLAMP, min(GRADE_CLAMP, (delta_e / d) * 100))
        legs.append(TrackLeg(prev.distance_m, curr.distance_m, d, grade, delta_e))
    if not legs: raise ValueError("Trasa je příliš krátká pro segmentaci.")
    return legs, asc, desc

def create_segments(legs: List[TrackLeg]) -> List[TrackLeg]:
    segs, cur_len, cur_grade, start = [], 0.0, 0.0, legs[0].start_distance
    for leg in legs:
        rem = leg.delta_distance
        while rem > 0:
            left = SEGMENT_LENGTH_METERS - cur_len
            take = min(left, rem)
            cur_grade += leg.grade_percent * take
            cur_len += take
            rem -= take
            if cur_len >= SEGMENT_LENGTH_METERS:
                avg = cur_grade / cur_len
                segs.append(TrackLeg(start, start + cur_len, cur_len, avg, avg / 100 * cur_len))
                start += cur_len; cur_len = 0.0; cur_grade = 0.0
    if cur_len > 1:
        avg = cur_grade / cur_len
        segs.append(TrackLeg(start, start + cur_len, cur_len, avg, avg / 100 * cur_len))
    return segs

def total_power(grade, speed, mass): 
    p_grav = mass * GRAVITY * speed * (grade / 100)
    p_roll = CRR * mass * GRAVITY * speed
    p_aero = 0.5 * RHO_AIR * CDA * speed**3
    return p_grav + p_roll + p_aero

def solve_speed_for_power(grade, target_power, mass):
    low, high = MIN_SPEED, MAX_SPEED
    def f(v): return total_power(grade, v, mass) - target_power
    for _ in range(60):
        mid = (low + high) / 2
        val = f(mid)
        if abs(val) < 0.5: return mid, False
        if val > 0: high = mid
        else: low = mid
    return (low + high) / 2, True

def coasting_speed(grade, mass): return solve_speed_for_power(grade, 0, mass)[0]


# --- Flask routes --------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    errors, results = [], None
    form_defaults = {"weight_rider": "", "weight_bike": "", "ftp": "", "mode": "speed"}

    if request.method == "POST":
        errors = validate_inputs(request.form)
        if not errors:
            gpx_file = request.files["gpx"]
            gpx_content = gpx_file.read().decode("utf-8")
            points = parse_gpx(gpx_content)
            legs, ascent, descent = build_legs(points)
            segs = create_segments(legs)

            results = calculate_segments(
                segments=segs,
                mass_total=float(request.form["weight_rider"]) + float(request.form["weight_bike"]),
                mode=request.form["mode"],
                ftp=float(request.form["ftp"]),
                total_ascent=ascent,
                total_descent=descent,
                target_speed_kmh=float(request.form.get("target_speed", 0)),
                target_rpe=int(request.form.get("target_rpe", 7)),
            )
        form_defaults = request.form

    return render_template("index.html", errors=errors, results=results, form_data=SimpleNamespace(**form_defaults))

@app.route("/save_result", methods=["POST"])
def save_result():
    user_id = request.form.get("user_id")
    result_data = request.form.get("result_data")
    if not user_id or not result_data:
        return {"error": "Missing data"}, 400
    db.collection("users").document(user_id).collection("rides").add({
        "data": result_data,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    return {"status": "ok"}

@app.route("/export.csv")
def export_csv() -> Response:
    if not LAST_CSV_ROWS:
        flash("Nejsou dostupná žádná data k exportu.")
        return redirect(url_for("index"))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(LAST_CSV_ROWS[0].keys()))
    writer.writeheader(); writer.writerows(LAST_CSV_ROWS)
    output.seek(0)
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=segments.csv"})

# --- Run -----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
