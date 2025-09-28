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

app = Flask(__name__)
app.secret_key = "dev-secret"  # In production use a proper secret key

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
    1: 0.35,
    2: 0.50,
    3: 0.60,
    4: 0.70,
    5: 0.80,
    6: 0.90,
    7: 1.00,
    8: 1.05,
    9: 1.15,
    10: 1.30,
}

# In-memory storage of last CSV rows (MVP only; not suitable for production).
LAST_CSV_ROWS: List[Dict[str, str]] = []


@dataclass
class GPXPoint:
    """Single GPX point enriched with cumulative distance."""

    lat: float
    lon: float
    ele: float
    distance_m: float


@dataclass
class TrackLeg:
    """Represents a small leg between two GPX points."""

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
    speed_kmh: List[float]
    target_power_line: Optional[List[float]] = None


@dataclass
class CalculationResult:
    segments: List[SegmentResult]
    summary: SummaryResult
    chart: ChartData
    mode: str
    target_power: Optional[float] = None
    target_speed_kmh: Optional[float] = None
    target_rpe: Optional[int] = None
    warning: Optional[str] = None


# --- Utility functions ---------------------------------------------------------

def format_time(seconds: float) -> str:
    """Return a mm:ss or hh:mm:ss formatted time string."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def moving_average(data: Iterable[float], window: int) -> np.ndarray:
    """Simple moving average with reflection at the edges."""

    arr = np.array(list(data), dtype=float)
    if len(arr) == 0:
        return arr
    if window <= 1:
        return arr
    window = window if window % 2 == 1 else window + 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def rpe_to_power(rpe: int, ftp: float) -> float:
    """Map RPE (1-10) to target power using predefined %FTP mapping."""

    return ftp * RPE_TO_PERCENT.get(rpe, 1.0)


def parse_gpx(gpx_content: str) -> List[GPXPoint]:
    """Parse GPX XML text into a list of GPXPoint with cumulative distance."""

    try:
        gpx = gpxpy.parse(gpx_content)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Soubor se nepodařilo načíst jako GPX.") from exc

    raw_points: List[Tuple[float, float, float]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.elevation is None:
                    continue
                raw_points.append((point.latitude, point.longitude, point.elevation))

    if not raw_points:
        raise ValueError("Trasa neobsahuje nadmořskou výšku nebo body.")

    # Remove consecutive duplicates and very small moves.
    filtered: List[Tuple[float, float, float]] = []
    last_lat, last_lon = None, None
    for lat, lon, ele in raw_points:
        if last_lat is None:
            filtered.append((lat, lon, ele))
            last_lat, last_lon = lat, lon
            continue
        distance = gpxpy.geo.haversine_distance(last_lat, last_lon, lat, lon)
        if distance is None or distance < 0.5:
            continue
        filtered.append((lat, lon, ele))
        last_lat, last_lon = lat, lon

    if len(filtered) < 2:
        raise ValueError("Trasa je příliš krátká pro segmentaci.")

    lats, lons, elevations = zip(*filtered)
    distances = [0.0]
    total = 0.0
    for idx in range(1, len(filtered)):
        dist = gpxpy.geo.haversine_distance(
            lats[idx - 1], lons[idx - 1], lats[idx], lons[idx]
        )
        if dist is None:
            dist = 0.0
        total += dist
        distances.append(total)

    if total < SEGMENT_LENGTH_METERS:
        raise ValueError("Trasa je příliš krátká pro segmentaci.")

    smoothed_elev = moving_average(elevations, SMOOTHING_WINDOW)

    points = [
        GPXPoint(lat=lat, lon=lon, ele=ele, distance_m=dist)
        for lat, lon, ele, dist in zip(lats, lons, smoothed_elev, distances)
    ]
    return points


def build_legs(points: List[GPXPoint]) -> Tuple[List[TrackLeg], float, float]:
    """Construct small legs between GPX points, computing grades and ascent/descent."""

    legs: List[TrackLeg] = []
    total_ascent = 0.0
    total_descent = 0.0
    for prev, current in zip(points, points[1:]):
        delta_dist = current.distance_m - prev.distance_m
        if delta_dist <= 0.5:
            continue
        delta_ele = current.ele - prev.ele
        if delta_ele > 0:
            total_ascent += delta_ele
        else:
            total_descent += abs(delta_ele)
        grade = (delta_ele / delta_dist) * 100 if delta_dist > 0 else 0.0
        grade = max(-GRADE_CLAMP, min(GRADE_CLAMP, grade))
        legs.append(
            TrackLeg(
                start_distance=prev.distance_m,
                end_distance=current.distance_m,
                delta_distance=delta_dist,
                grade_percent=grade,
                delta_elevation=delta_ele,
            )
        )
    if not legs:
        raise ValueError("Trasa je příliš krátká pro segmentaci.")
    return legs, total_ascent, total_descent


def create_segments(legs: List[TrackLeg]) -> List[TrackLeg]:
    """Aggregate legs into fixed-length segments."""

    segments: List[TrackLeg] = []
    current_length = 0.0
    current_weighted_grade = 0.0
    seg_start = legs[0].start_distance
    for leg in legs:
        remaining_leg = leg.delta_distance
        leg_grade = leg.grade_percent
        while remaining_leg > 0:
            space_left = SEGMENT_LENGTH_METERS - current_length
            take = min(space_left, remaining_leg)
            current_weighted_grade += leg_grade * take
            current_length += take
            remaining_leg -= take
            if current_length + 1e-6 >= SEGMENT_LENGTH_METERS:
                avg_grade = current_weighted_grade / current_length if current_length else 0.0
                segments.append(
                    TrackLeg(
                        start_distance=seg_start,
                        end_distance=seg_start + current_length,
                        delta_distance=current_length,
                        grade_percent=avg_grade,
                        delta_elevation=avg_grade / 100.0 * current_length,
                    )
                )
                seg_start += current_length
                current_length = 0.0
                current_weighted_grade = 0.0
        # If leg ends but segment not full, continue with next leg

    if current_length > 1.0:
        avg_grade = current_weighted_grade / current_length if current_length else 0.0
        segments.append(
            TrackLeg(
                start_distance=seg_start,
                end_distance=seg_start + current_length,
                delta_distance=current_length,
                grade_percent=avg_grade,
                delta_elevation=avg_grade / 100.0 * current_length,
            )
        )

    return segments


def total_power(grade_percent: float, speed_mps: float, mass_total: float) -> float:
    """Calculate total power demand for given grade and speed."""

    theta_component = grade_percent / 100.0
    p_grav = mass_total * GRAVITY * speed_mps * theta_component
    p_roll = CRR * mass_total * GRAVITY * speed_mps
    p_aero = 0.5 * RHO_AIR * CDA * speed_mps**3
    return p_grav + p_roll + p_aero


def speed_from_power(
    target_power: float, grade_percent: float, mass_total: float
) -> Tuple[float, bool]:
    """Compute steady-state speed for a given target power using bisection.

    The search follows the simplified cycling power model and clamps the
    solution to the interval ``[MIN_SPEED, MAX_SPEED]``. If the requested power
    cannot be reached inside that interval (for example on a very steep climb or
    descent), the closest boundary value is returned and ``limited`` is marked
    as ``True``.
    """

    low, high = MIN_SPEED, MAX_SPEED

    def total_difference(speed: float) -> float:
        return total_power(grade_percent, speed, mass_total) - target_power

    power_low = total_difference(low)
    power_high = total_difference(high)

    # Target power is below what is possible even at the slowest allowed speed.
    if power_low >= 0:
        return low, True

    # Target power is above what is possible even at the fastest allowed speed.
    if power_high <= 0:
        return high, True

    limited = False
    for _ in range(100):
        mid = (low + high) / 2
        diff = total_difference(mid)
        if abs(diff) <= 0.5:
            return mid, limited
        if diff > 0:
            high = mid
        else:
            low = mid

    # Max iterations reached; mark as limited and return the midpoint.
    limited = True
    return (low + high) / 2, limited


def coasting_speed(grade_percent: float, mass_total: float) -> float:
    """Speed achieved when target power is zero (coasting)."""

    speed, _ = speed_from_power(0.0, grade_percent, mass_total)
    return speed


def calculate_segments(
    segments: List[TrackLeg],
    mass_total: float,
    mode: str,
    ftp: float,
    total_ascent: float,
    total_descent: float,
    target_speed_kmh: Optional[float] = None,
    target_rpe: Optional[int] = None,
) -> CalculationResult:
    """Run the physics model on segments and prepare formatted results."""

    total_distance = sum(seg.delta_distance for seg in segments)
    segment_results: List[SegmentResult] = []
    chart_distance = []
    chart_power = []
    chart_grade = []
    chart_speed = []
    warning: Optional[str] = None
    target_power: Optional[float] = None
    target_speed: Optional[float] = None
    target_rpe_value: Optional[int] = None
    total_time = 0.0

    if mode == "speed":
        assert target_speed_kmh is not None
        speed_mps = target_speed_kmh / 3.6
        target_speed = target_speed_kmh

    elif mode == "rpe":
        assert target_rpe is not None
        target_rpe_value = target_rpe
        target_power = rpe_to_power(target_rpe, ftp)
    else:
        raise ValueError("Neznámý režim výpočtu.")

    for idx, seg in enumerate(segments, start=1):
        grade = seg.grade_percent
        power_display = ""
        if mode == "speed":
            power = total_power(grade, speed_mps, mass_total)
            limited = False
            if power < 0:
                power = 0.0
                speed_calc = coasting_speed(grade, mass_total)
                speed_mps_eff = max(speed_calc, MIN_SPEED)
                power_display = f"0 (volnoběh {speed_mps_eff * 3.6:.1f} km/h)"
            else:
                speed_mps_eff = speed_mps
            time_seg = seg.delta_distance / speed_mps_eff
            speed_kmh = speed_mps_eff * 3.6
            if not power_display:
                power_display = f"{power:.0f}"
        else:  # RPE mode
            speed_mps_eff, limited = speed_from_power(target_power, grade, mass_total)
            power = max(target_power, 0.0)
            time_seg = seg.delta_distance / speed_mps_eff
            speed_kmh = speed_mps_eff * 3.6
            speed_note = " (limit)" if limited else ""
            power_display = f"{power:.0f}"

        total_time += time_seg
        chart_distance.append(round(seg.end_distance / 1000.0, 3))
        chart_power.append(round(power, 2))
        chart_grade.append(round(grade, 2))
        chart_speed.append(round(speed_kmh, 2))

        segment_results.append(
            SegmentResult(
                index=idx,
                km_from=f"{seg.start_distance / 1000:.1f}",
                km_to=f"{seg.end_distance / 1000:.1f}",
                length_m=f"{seg.delta_distance:.0f}",
                grade_percent=f"{grade:.1f}",
                power_w=power_display,
                speed_kmh=f"{speed_kmh:.1f}" + (speed_note if mode == "rpe" else ""),
                time_str=format_time(time_seg),
            )
        )

    avg_speed_kmh = (total_distance / 1000) / (total_time / 3600) if total_time > 0 else 0

    summary = SummaryResult(
        total_distance_km=f"{total_distance / 1000:.1f}",
        total_ascent=f"{total_ascent:.0f}",
        total_descent=f"{total_descent:.0f}",
        avg_speed_kmh=f"{avg_speed_kmh:.1f}",
        total_time=format_time(total_time),
    )

    if mode == "rpe" and target_rpe_value is not None and target_rpe_value >= 9 and total_time > 1800:
        warning = "RPE 9–10 jsou udržitelné jen krátce. Výsledek nemusí být realistický."

    target_power_line = [target_power] * len(chart_distance) if target_power is not None else None

    chart = ChartData(
        distance_km=chart_distance,
        power_w=chart_power,
        grade_percent=chart_grade,
        speed_kmh=chart_speed,
        target_power_line=target_power_line,
    )

    # Prepare CSV rows for export
    LAST_CSV_ROWS.clear()
    for seg in segment_results:
        LAST_CSV_ROWS.append(
            {
                "index": seg.index,
                "km_from": seg.km_from,
                "km_to": seg.km_to,
                "length_m": seg.length_m,
                "grade_percent": seg.grade_percent,
                "power_w": seg.power_w,
                "speed_kmh": seg.speed_kmh,
                "time": seg.time_str,
            }
        )

    return CalculationResult(
        segments=segment_results,
        summary=summary,
        chart=chart,
        mode=mode,
        target_power=target_power,
        target_speed_kmh=target_speed,
        target_rpe=target_rpe_value,
        warning=warning,
    )


# --- Flask routes --------------------------------------------------------------


def validate_inputs(form: Dict[str, str]) -> List[str]:
    """Validate form inputs and return a list of error messages."""

    errors: List[str] = []
    try:
        weight_rider = float(form.get("weight_rider", ""))
        if weight_rider <= 0:
            errors.append("Hmotnost jezdce musí být kladné číslo.")
    except ValueError:
        errors.append("Neplatná hmotnost jezdce.")

    try:
        weight_bike = float(form.get("weight_bike", ""))
        if weight_bike <= 0:
            errors.append("Hmotnost kola musí být kladné číslo.")
    except ValueError:
        errors.append("Neplatná hmotnost kola.")

    try:
        ftp = float(form.get("ftp", ""))
        if ftp <= 0:
            errors.append("FTP musí být kladné číslo.")
    except ValueError:
        errors.append("Neplatné FTP.")

    mode = form.get("mode", "speed")
    if mode == "speed":
        try:
            target_speed = float(form.get("target_speed", ""))
            if not (5 <= target_speed <= 60):
                errors.append("Cílová rychlost musí být mezi 5 a 60 km/h.")
        except ValueError:
            errors.append("Neplatná cílová rychlost.")
    elif mode == "rpe":
        try:
            rpe = int(form.get("target_rpe", ""))
            if not (1 <= rpe <= 10):
                errors.append("RPE musí být v rozmezí 1–10.")
        except ValueError:
            errors.append("Neplatné RPE.")
    else:
        errors.append("Neznámý režim výpočtu.")

    return errors


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    errors: List[str] = []
    results: Optional[CalculationResult] = None
    form_defaults = {
        "weight_rider": request.form.get("weight_rider"),
        "weight_bike": request.form.get("weight_bike"),
        "ftp": request.form.get("ftp"),
        "mode": request.form.get("mode", "speed"),
        "target_speed": request.form.get("target_speed"),
        "target_rpe": request.form.get("target_rpe"),
    }

    if request.method == "POST":
        errors = validate_inputs(request.form)
        gpx_file = request.files.get("gpx")
        if not gpx_file or gpx_file.filename == "":
            errors.append("Musíte nahrát GPX soubor.")

        if not errors and gpx_file:
            try:
                raw_bytes = gpx_file.read()
                points = parse_gpx(raw_bytes.decode("utf-8", errors="ignore"))
                legs, total_ascent, total_descent = build_legs(points)
                segments = create_segments(legs)
                mass_total = float(request.form["weight_rider"]) + float(request.form["weight_bike"])
                ftp = float(request.form["ftp"])
                mode = request.form.get("mode", "speed")
                if mode == "speed":
                    target_speed = float(request.form["target_speed"])
                    results = calculate_segments(
                        segments=segments,
                        mass_total=mass_total,
                        mode=mode,
                        ftp=ftp,
                        total_ascent=total_ascent,
                        total_descent=total_descent,
                        target_speed_kmh=target_speed,
                    )
                else:
                    target_rpe = int(request.form["target_rpe"])
                    results = calculate_segments(
                        segments=segments,
                        mass_total=mass_total,
                        mode=mode,
                        ftp=ftp,
                        total_ascent=total_ascent,
                        total_descent=total_descent,
                        target_rpe=target_rpe,
                    )
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                errors.append("Během zpracování došlo k chybě.")
                app.logger.exception("Unexpected error", exc_info=exc)

    return render_template(
        "index.html",
        errors=errors,
        results=results,
        form_data=SimpleNamespace(**form_defaults),
    )


@app.route("/export.csv")
def export_csv() -> Response:
    if not LAST_CSV_ROWS:
        flash("Nejsou dostupná žádná data k exportu.")
        return redirect(url_for("index"))

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "index",
            "km_from",
            "km_to",
            "length_m",
            "grade_percent",
            "power_w",
            "speed_kmh",
            "time",
        ],
    )
    writer.writeheader()
    writer.writerows(LAST_CSV_ROWS)
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=segments.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)
