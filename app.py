"""Flask application serving a simple cycling power/pacing calculator."""
from __future__ import annotations

import csv
import io
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import gpxpy
import gpxpy.gpx
import numpy as np
from flask import Flask, Response, abort, flash, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf import CSRFProtect
import bcrypt
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError

from forms import LoginForm, RegisterForm
from models import Training, User, db

try:
    from fitparse import FitFile
except ImportError:  # pragma: no cover - dependency should be installed via requirements
    FitFile = None  # type: ignore[assignment]

app = Flask(__name__)
app.secret_key = "dev-secret"  # In production use a proper secret key
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cycling.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
csrf = CSRFProtect(app)

# Ensure upload directory exists in the Flask instance folder.
UPLOAD_FOLDER = Path(app.instance_path) / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    """Return the authenticated user for Flask-Login."""

    if user_id and user_id.isdigit():
        return db.session.get(User, int(user_id))
    return None


with app.app_context():
    db.create_all()

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


@dataclass
class ActivityPoint:
    """Represents a single data point from a workout file."""

    time: Optional[datetime]
    lat: Optional[float]
    lon: Optional[float]
    ele: Optional[float]
    heart_rate: Optional[float]
    power: Optional[float]
    speed: Optional[float]


@dataclass
class ActivitySummaryDisplay:
    total_time: str
    distance_km: str
    elevation_gain: str
    elevation_loss: str
    avg_speed: str
    max_speed: str
    avg_hr: str
    max_hr: str
    avg_power: str
    max_power: str
    normalized_power: str
    intensity_factor: str
    training_stress_score: str
    calories: str
    variability_index: str


@dataclass
class ActivityAnalysisData:
    stats: "ActivityStats"
    summary: ActivitySummaryDisplay
    coords: List[List[float]]
    time_labels: List[str]
    dist_labels: List[float]
    hr_values: List[Optional[float]]
    power_values: List[Optional[float]]
    speed_values: List[Optional[float]]
    elev_values: List[Optional[float]]


@dataclass
class ActivityStats:
    start_time: Optional[datetime]
    total_seconds: float
    distance_km: float
    elevation_gain: float
    elevation_loss: float
    avg_speed: Optional[float]
    max_speed: Optional[float]
    avg_hr: Optional[float]
    max_hr: Optional[float]
    avg_power: Optional[float]
    max_power: Optional[float]
    normalized_power: Optional[float]
    intensity_factor: Optional[float]
    training_stress_score: Optional[float]
    calories: float
    variability_index: Optional[float]

# --- Utility functions ---------------------------------------------------------

def format_time(seconds: float) -> str:
    """Return a mm:ss or hh:mm:ss formatted time string."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_hhmmss(seconds: float) -> str:
    """Format elapsed time in HH:MM:SS regardless of duration."""

    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


app.jinja_env.globals.update(format_time=format_time, format_hhmmss=format_hhmmss)


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


def iso_week_start(dt: datetime) -> date:
    """Return the Monday date for the ISO week of the provided datetime."""

    iso_weekday = dt.isoweekday()
    return dt.date() - timedelta(days=iso_weekday - 1)


def build_weekly_summaries(trainings: Iterable[Training]) -> List[Dict[str, str]]:
    """Aggregate total time, distance, and TSS per ISO week."""

    weekly: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for training in trainings:
        if training.date is None:
            continue
        iso_year, iso_week, _ = training.date.isocalendar()
        week_start = iso_week_start(training.date)
        bucket = weekly.setdefault(
            (iso_year, iso_week),
            {"seconds": 0.0, "distance": 0.0, "tss": 0.0, "start": week_start},
        )
        bucket["seconds"] += float(training.duration or 0.0)
        bucket["distance"] += float(training.distance or 0.0)
        bucket["tss"] += float(training.training_stress_score or 0.0)

    summaries: List[Dict[str, str]] = []
    for data in sorted(weekly.values(), key=lambda item: item["start"], reverse=True):
        start_date = data["start"]
        end_date = start_date + timedelta(days=6)
        summaries.append(
            {
                "label": f"{start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}",
                "time": format_hhmmss(data["seconds"]),
                "distance": f"{data['distance']:.1f}",
                "tss": f"{data['tss']:.0f}",
            }
        )

    return summaries


def build_calendar_events(trainings: Iterable[Training]) -> List[Dict[str, object]]:
    """Prepare FullCalendar event structures for the user's trainings."""

    events: List[Dict[str, object]] = []
    for training in trainings:
        if training.date is None:
            continue
        distance = float(training.distance or 0.0)
        tss = float(training.training_stress_score or 0.0)
        title = training.title or "Trénink"
        events.append(
            {
                "id": training.id,
                "title": f"{title} ({distance:.1f} km, TSS {tss:.0f})",
                "start": training.date.isoformat(),
                "url": url_for("training_detail", training_id=training.id),
            }
        )
    return events


def build_performance_series(trainings: Iterable[Training]) -> Dict[str, List[float]]:
    """Compute CTL/ATL/TSB daily series using the Banister model."""

    dated_trainings = [t for t in trainings if t.date is not None]
    if not dated_trainings:
        return {"labels": [], "ctl": [], "atl": [], "tsb": []}

    dated_trainings.sort(key=lambda tr: tr.date)
    daily_tss: Dict[date, float] = defaultdict(float)
    for training in dated_trainings:
        training_date = training.date.date()
        daily_tss[training_date] += float(training.training_stress_score or 0.0)

    start_date = dated_trainings[0].date.date()
    end_date = max(date.today(), dated_trainings[-1].date.date())

    labels: List[str] = []
    ctl_values: List[float] = []
    atl_values: List[float] = []
    tsb_values: List[float] = []

    ctl = 0.0
    atl = 0.0
    current_day = start_date
    while current_day <= end_date:
        tss_today = daily_tss.get(current_day, 0.0)
        ctl += (tss_today - ctl) / 42.0
        atl += (tss_today - atl) / 7.0
        labels.append(current_day.strftime("%Y-%m-%d"))
        ctl_values.append(round(ctl, 2))
        atl_values.append(round(atl, 2))
        tsb_values.append(round(ctl - atl, 2))
        current_day += timedelta(days=1)

    return {"labels": labels, "ctl": ctl_values, "atl": atl_values, "tsb": tsb_values}


def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 timestamps from TCX/FIT files into timezone-aware datetimes."""

    if not value:
        return None
    text = value.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def to_utc(dt: datetime) -> datetime:
    """Normalise datetime to UTC for safe subtraction."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Convert text to float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_text_with_suffix(element: ET.Element, suffix: str) -> Optional[str]:
    """Search for the first descendant tag ending with the provided suffix."""

    suffix_lower = suffix.lower()
    for child in element.iter():
        if child.tag.lower().endswith(suffix_lower) and child.text:
            return child.text
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two WGS84 coordinates in meters."""

    radius = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def rpe_to_power(rpe: int, ftp: float) -> float:
    """Map RPE (1-10) to target power using predefined %FTP mapping."""

    return ftp * RPE_TO_PERCENT.get(rpe, 1.0)


def compute_normalized_power(power_time_pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Calculate Normalized Power from (time, power) samples."""

    if len(power_time_pairs) < 2:
        return None

    # Normalise time so that the first sample starts at zero seconds.
    base_time = power_time_pairs[0][0]
    times = np.array([t - base_time for t, _ in power_time_pairs], dtype=float)
    powers = np.array([p for _, p in power_time_pairs], dtype=float)

    if times[-1] <= 0:
        return None

    # Resample to 1-second resolution using linear interpolation to create
    # a smooth time series that the 30-second rolling mean can be applied to.
    seconds = np.arange(0, math.ceil(times[-1]) + 1, dtype=float)
    interpolated = np.interp(seconds, times, powers)

    window = 30
    if len(interpolated) < window:
        rolling = np.array([interpolated.mean()])
    else:
        kernel = np.ones(window) / window
        rolling = np.convolve(interpolated, kernel, mode="valid")

    fourth_power_mean = np.mean(np.power(rolling, 4))
    return float(fourth_power_mean ** 0.25) if fourth_power_mean > 0 else None


def compute_work_joules(power_time_pairs: List[Tuple[float, float]]) -> float:
    """Integrate power over time to get total work in joules."""

    if len(power_time_pairs) < 2:
        return 0.0

    work = 0.0
    for (t0, p0), (t1, p1) in zip(power_time_pairs[:-1], power_time_pairs[1:]):
        delta_t = max(0.0, t1 - t0)
        if delta_t == 0:
            continue
        # Use trapezoidal rule with average power between the two samples.
        work += (p0 + p1) / 2.0 * delta_t
    return work


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


def parse_tcx_workout(content: bytes) -> List[ActivityPoint]:
    """Parse a TCX workout file and extract activity points."""

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("Soubor se nepodařilo načíst jako TCX.") from exc

    points: List[ActivityPoint] = []
    for tp in root.iter():
        if not tp.tag.lower().endswith("trackpoint"):
            continue
        time_text = _find_text_with_suffix(tp, "Time")
        lat = _safe_float(_find_text_with_suffix(tp, "LatitudeDegrees"))
        lon = _safe_float(_find_text_with_suffix(tp, "LongitudeDegrees"))
        ele = _safe_float(_find_text_with_suffix(tp, "AltitudeMeters"))

        hr_value = None
        for child in tp.iter():
            if child.tag.lower().endswith("heartratebpm"):
                hr_value = _find_text_with_suffix(child, "Value") or child.text
                break

        point = ActivityPoint(
            time=parse_iso8601(time_text),
            lat=lat,
            lon=lon,
            ele=ele,
            heart_rate=_safe_float(hr_value),
            power=_safe_float(_find_text_with_suffix(tp, "Watts")),
            speed=_safe_float(_find_text_with_suffix(tp, "Speed")),
        )

        if not any(
            field is not None
            for field in (point.time, point.lat, point.lon, point.ele, point.heart_rate, point.power, point.speed)
        ):
            continue
        points.append(point)

    if not points:
        raise ValueError("Soubor neobsahuje žádná použitelná data.")
    return points


def parse_fit_workout(content: bytes) -> List[ActivityPoint]:
    """Parse a FIT workout file using fitparse."""

    if FitFile is None:
        raise ValueError("Knihovna fitparse není dostupná. Zkontrolujte instalaci závislostí.")

    try:
        fit = FitFile(io.BytesIO(content))
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Soubor se nepodařilo načíst jako FIT.") from exc

    points: List[ActivityPoint] = []
    try:
        for record in fit.get_messages("record"):
            data = {field.name: field.value for field in record}
            lat_raw = data.get("position_lat")
            lon_raw = data.get("position_long")
            lat_deg = lat_raw * (180 / 2**31) if lat_raw is not None else None
            lon_deg = lon_raw * (180 / 2**31) if lon_raw is not None else None

            point = ActivityPoint(
                time=data.get("timestamp"),
                lat=lat_deg,
                lon=lon_deg,
                ele=data.get("altitude"),
                heart_rate=data.get("heart_rate"),
                power=data.get("power"),
                speed=data.get("speed"),
            )

            if not any(
                field is not None
                for field in (point.time, point.lat, point.lon, point.ele, point.heart_rate, point.power, point.speed)
            ):
                continue
            points.append(point)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Během čtení FIT souboru došlo k chybě.") from exc

    if not points:
        raise ValueError("Soubor neobsahuje žádná použitelná data.")
    return points


def build_activity_analysis(points: List[ActivityPoint], ftp: float = 250.0) -> ActivityAnalysisData:
    """Aggregate workout points into summary metrics and chart-friendly series."""

    if not points:
        raise ValueError("Soubor neobsahuje žádná data.")

    map_points = [
        (p.lat, p.lon)
        for p in points
        if p.lat is not None and p.lon is not None
    ]
    if len(map_points) < 2:
        raise ValueError("Soubor neobsahuje GPS data.")

    total_distance = 0.0
    total_ascent = 0.0
    total_descent = 0.0
    cumulative_distances = [0.0]

    heart_rate_values: List[float] = []
    power_values: List[float] = []
    speed_values: List[float] = []
    elevation_values: List[float] = []
    power_time_pairs: List[Tuple[float, float]] = []

    hr_series: List[Optional[float]] = []
    power_series: List[Optional[float]] = []
    speed_series: List[Optional[float]] = []
    elevation_series: List[Optional[float]] = []

    start_time = next((p.time for p in points if p.time is not None), None)
    elapsed_seconds: List[float] = []
    last_elapsed = 0.0

    if start_time is not None:
        start_time_utc = to_utc(start_time)
        for point in points:
            if point.time is not None:
                last_elapsed = max(0.0, (to_utc(point.time) - start_time_utc).total_seconds())
            elapsed_seconds.append(last_elapsed)
    else:
        for idx, _ in enumerate(points):
            elapsed_seconds.append(float(idx))

    prev_point = points[0]
    for point in points[1:]:
        dist = 0.0
        if (
            prev_point.lat is not None
            and prev_point.lon is not None
            and point.lat is not None
            and point.lon is not None
        ):
            dist = haversine_distance(prev_point.lat, prev_point.lon, point.lat, point.lon)
        total_distance += dist
        cumulative_distances.append(total_distance)

        if prev_point.ele is not None and point.ele is not None:
            delta_ele = point.ele - prev_point.ele
            if delta_ele > 0:
                total_ascent += delta_ele
            elif delta_ele < 0:
                total_descent += abs(delta_ele)

        prev_point = point

    for idx, point in enumerate(points):
        hr_value = float(point.heart_rate) if point.heart_rate is not None else None
        power_value = float(point.power) if point.power is not None else None
        speed_value = float(point.speed) * 3.6 if point.speed is not None else None
        elevation_value = float(point.ele) if point.ele is not None else None

        hr_series.append(hr_value)
        power_series.append(power_value)
        speed_series.append(speed_value)
        elevation_series.append(elevation_value)

        if hr_value is not None:
            heart_rate_values.append(hr_value)
        if power_value is not None:
            power_values.append(power_value)
            power_time_pairs.append((elapsed_seconds[idx], power_value))
        if speed_value is not None:
            speed_values.append(speed_value)
        if elevation_value is not None:
            elevation_values.append(elevation_value)

        if len(cumulative_distances) <= idx:
            cumulative_distances.append(cumulative_distances[-1])

    total_time_seconds = elapsed_seconds[-1] if elapsed_seconds else 0.0
    total_distance_km = total_distance / 1000.0

    avg_speed = (total_distance / total_time_seconds * 3.6) if total_time_seconds > 0 else None
    max_speed = max(speed_values) if speed_values else None
    avg_hr = sum(heart_rate_values) / len(heart_rate_values) if heart_rate_values else None
    max_hr = max(heart_rate_values) if heart_rate_values else None
    avg_power = sum(power_values) / len(power_values) if power_values else None
    max_power = max(power_values) if power_values else None

    normalized_power = compute_normalized_power(power_time_pairs) if power_time_pairs else None
    intensity_factor = (normalized_power / ftp) if normalized_power and ftp > 0 else None
    variability_index = (normalized_power / avg_power) if normalized_power and avg_power else None
    total_work_j = compute_work_joules(power_time_pairs) if power_time_pairs else 0.0
    calories = total_work_j * 0.000239
    training_stress_score = None
    if normalized_power and intensity_factor and ftp > 0 and total_time_seconds > 0:
        training_stress_score = (
            (total_time_seconds * normalized_power * intensity_factor) / (ftp * 3600.0) * 100.0
        )

    elevation_gain_str = f"{total_ascent:.0f}" if elevation_values else "—"
    elevation_loss_str = f"{total_descent:.0f}" if elevation_values else "—"

    summary = ActivitySummaryDisplay(
        total_time=format_time(total_time_seconds) if total_time_seconds > 0 else "—",
        distance_km=f"{total_distance_km:.1f}",
        elevation_gain=elevation_gain_str,
        elevation_loss=elevation_loss_str,
        avg_speed=f"{avg_speed:.1f}" if avg_speed is not None else "—",
        max_speed=f"{max_speed:.1f}" if max_speed is not None else "—",
        avg_hr=f"{avg_hr:.0f}" if avg_hr is not None else "—",
        max_hr=f"{max_hr:.0f}" if max_hr is not None else "—",
        avg_power=f"{avg_power:.0f}" if avg_power is not None else "—",
        max_power=f"{max_power:.0f}" if max_power is not None else "—",
        normalized_power=f"{normalized_power:.0f}" if normalized_power is not None else "—",
        intensity_factor=f"{intensity_factor:.2f}" if intensity_factor is not None else "—",
        training_stress_score=f"{training_stress_score:.0f}" if training_stress_score is not None else "—",
        calories=f"{calories:.0f}",
        variability_index=f"{variability_index:.2f}" if variability_index is not None else "—",
    )

    stats = ActivityStats(
        start_time=start_time,
        total_seconds=total_time_seconds,
        distance_km=total_distance_km,
        elevation_gain=total_ascent,
        elevation_loss=total_descent,
        avg_speed=avg_speed,
        max_speed=max_speed,
        avg_hr=avg_hr,
        max_hr=max_hr,
        avg_power=avg_power,
        max_power=max_power,
        normalized_power=normalized_power,
        intensity_factor=intensity_factor,
        training_stress_score=training_stress_score,
        calories=calories,
        variability_index=variability_index,
    )

    return ActivityAnalysisData(
        stats=stats,
        summary=summary,
        coords=[[float(lat), float(lon)] for lat, lon in map_points],
        time_labels=[format_hhmmss(seconds) for seconds in elapsed_seconds],
        dist_labels=[round(distance / 1000.0, 3) for distance in cumulative_distances],
        hr_values=hr_series,
        power_values=power_series,
        speed_values=speed_series,
        elev_values=elevation_series,
    )


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


@app.route("/")
def hero() -> str:
    return render_template("hero.html")


@app.route("/register", methods=["GET", "POST"])
def register() -> str:
    if current_user.is_authenticated:
        return redirect(url_for("hero"))

    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        email = form.email.data.lower()
        password = form.password.data
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = User(username=username, email=email, password_hash=hashed)
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("Uživatel s tímto e-mailem nebo jménem již existuje.", "danger")
        else:
            login_user(user)
            flash("Registrace proběhla úspěšně.", "success")
            return redirect(url_for("hero"))

    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    if current_user.is_authenticated:
        return redirect(url_for("hero"))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.lower()
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(
            form.password.data.encode("utf-8"), user.password_hash.encode("utf-8")
        ):
            login_user(user)
            flash("Přihlášení proběhlo úspěšně.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("hero"))
        flash("Neplatné přihlašovací údaje.", "danger")

    return render_template("login.html", form=form)


@app.route("/logout")
@login_required
def logout() -> Response:
    logout_user()
    flash("Byli jste odhlášeni.", "info")
    return redirect(url_for("hero"))


@app.route("/calculator", methods=["GET", "POST"])
def calculator() -> str:
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


@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis() -> str:
    errors: List[str] = []
    ftp_default = request.form.get("ftp", "")
    title_default = request.form.get("title", "")

    if request.method == "POST":
        upload = request.files.get("workout")
        ftp_text = request.form.get("ftp", "")
        title = request.form.get("title", "").strip()
        ftp_value: Optional[float] = None

        try:
            ftp_value = float(ftp_text) if ftp_text else 250.0
            if ftp_value <= 0:
                raise ValueError
        except ValueError:
            errors.append("FTP musí být kladné číslo.")

        if not upload or upload.filename == "":
            errors.append("Musíte nahrát TCX nebo FIT soubor.")

        if not errors and upload and ftp_value is not None:
            file_bytes = upload.read()
            if not file_bytes:
                errors.append("Soubor je prázdný.")
            else:
                ext = os.path.splitext(upload.filename.lower())[-1]
                points: Optional[List[ActivityPoint]] = None
                try:
                    if ext == ".tcx":
                        points = parse_tcx_workout(file_bytes)
                    elif ext == ".fit":
                        points = parse_fit_workout(file_bytes)
                    else:
                        errors.append("Nepodporovaný formát. Nahrajte TCX nebo FIT soubor.")

                    if points is not None and not errors:
                        activity = build_activity_analysis(points, ftp=ftp_value)
                        activity_start = activity.stats.start_time
                        if activity_start is not None:
                            activity_start = to_utc(activity_start).replace(tzinfo=None)
                        else:
                            activity_start = datetime.utcnow()
                        default_title = f"Trénink {activity_start.strftime('%Y-%m-%d %H:%M')}"
                        safe_title = title or default_title

                        filename = secure_filename(upload.filename or "training.tcx")
                        if not filename:
                            filename = f"training{ext or '.tcx'}"
                        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                        storage_path = UPLOAD_FOLDER / f"{timestamp}_{filename}"
                        storage_path.write_bytes(file_bytes)

                        training = Training(
                            user=current_user,
                            date=activity_start,
                            title=safe_title,
                            distance=activity.stats.distance_km,
                            duration=activity.stats.total_seconds,
                            elevation=activity.stats.elevation_gain,
                            elevation_loss=activity.stats.elevation_loss,
                            avg_speed=activity.stats.avg_speed,
                            max_speed=activity.stats.max_speed,
                            avg_hr=activity.stats.avg_hr,
                            max_hr=activity.stats.max_hr,
                            avg_power=activity.stats.avg_power,
                            max_power=activity.stats.max_power,
                            normalized_power=activity.stats.normalized_power,
                            intensity_factor=activity.stats.intensity_factor,
                            training_stress_score=activity.stats.training_stress_score,
                            variability_index=activity.stats.variability_index,
                            calories=activity.stats.calories,
                            ftp_used=ftp_value,
                            file_path=str(storage_path),
                        )
                        db.session.add(training)
                        db.session.commit()
                        flash("Trénink byl úspěšně analyzován a uložen.", "success")
                        return redirect(url_for("training_detail", training_id=training.id))
                except ValueError as exc:
                    db.session.rollback()
                    errors.append(str(exc))
                except Exception as exc:  # pylint: disable=broad-except
                    db.session.rollback()
                    errors.append("Během zpracování došlo k chybě.")
                    app.logger.exception("Chyba při analýze tréninku", exc_info=exc)

    return render_template(
        "analysis.html",
        errors=errors,
        ftp_value=ftp_default,
        title_value=title_default,
    )


@app.route("/dashboard")
@login_required
def dashboard() -> str:
    trainings = (
        Training.query.filter_by(user_id=current_user.id)
        .order_by(Training.date.asc())
        .all()
    )
    calendar_events = build_calendar_events(trainings)
    weekly_summaries = build_weekly_summaries(trainings)[:6]
    performance_series = build_performance_series(trainings)

    return render_template(
        "dashboard.html",
        trainings=trainings,
        calendar_events=calendar_events,
        weekly_summaries=weekly_summaries,
        performance=performance_series,
    )


@app.route("/training/<int:training_id>")
@login_required
def training_detail(training_id: int) -> str:
    training = Training.query.filter_by(id=training_id, user_id=current_user.id).first()
    if training is None:
        abort(404)

    file_path = Path(training.file_path)
    if not file_path.exists():
        flash("Původní soubor se nepodařilo najít.", "warning")
        return redirect(url_for("dashboard"))

    file_bytes = file_path.read_bytes()
    ext = file_path.suffix.lower()
    points: Optional[List[ActivityPoint]] = None
    try:
        if ext == ".tcx":
            points = parse_tcx_workout(file_bytes)
        elif ext == ".fit":
            points = parse_fit_workout(file_bytes)
        else:
            flash("Formát souboru již není podporován.", "danger")
            return redirect(url_for("dashboard"))
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("dashboard"))

    if not points:
        flash("Soubor neobsahuje žádná data.", "danger")
        return redirect(url_for("dashboard"))

    ftp_value = training.ftp_used or 250.0
    analysis_data = build_activity_analysis(points, ftp=ftp_value)

    return render_template(
        "training_detail.html",
        training=training,
        analysis=analysis_data,
        ftp=ftp_value,
    )


@app.route("/export.csv")
def export_csv() -> Response:
    if not LAST_CSV_ROWS:
        flash("Nejsou dostupná žádná data k exportu.")
        return redirect(url_for("calculator"))

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
