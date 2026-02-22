from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
load_dotenv()

app = FastAPI(title="Ironman Coach API", version="2.0.0")

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

STRAVA_BASE = "https://www.strava.com/api/v3"


def _require_env() -> None:
    missing = []
    if not STRAVA_CLIENT_ID:
        missing.append("STRAVA_CLIENT_ID")
    if not STRAVA_CLIENT_SECRET:
        missing.append("STRAVA_CLIENT_SECRET")
    if not STRAVA_REFRESH_TOKEN:
        missing.append("STRAVA_REFRESH_TOKEN")
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing environment variables: {', '.join(missing)}",
        )


def get_access_token() -> str:
    """
    Exchanges refresh token for a new access token.
    """
    _require_env()
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "refresh_token": STRAVA_REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {r.text}")
    data = r.json()
    if "access_token" not in data:
        raise HTTPException(status_code=500, detail=f"Token refresh missing access_token: {data}")
    return data["access_token"]


def strava_get(path: str, token: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Generic Strava GET helper.
    """
    url = f"{STRAVA_BASE}{path}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params or {},
        timeout=30,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Strava fetch failed ({path}): {r.text}")
    return r.json()


# ------------------------------------------------------------
# Unit helpers
# ------------------------------------------------------------
def meters_to_miles(m: float) -> float:
    return m / 1609.344


def meters_to_km(m: float) -> float:
    return m / 1000.0


def meters_to_feet(m: float) -> float:
    return m * 3.280839895


def seconds_to_hhmmss(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def speed_mps_to_min_per_mile(mps: float) -> Optional[str]:
    if mps is None or mps <= 0:
        return None
    # pace = time / distance
    sec_per_mile = 1609.344 / mps
    mm = int(sec_per_mile // 60)
    ss = int(round(sec_per_mile - mm * 60))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm}:{ss:02d}"


# ------------------------------------------------------------
# Downsampling + aggregation
# ------------------------------------------------------------
def downsample_series(arr: List[Any], max_points: int) -> List[Any]:
    """
    Simple downsample by taking every Nth element.
    """
    if not arr:
        return []
    if max_points <= 0:
        return arr
    n = len(arr)
    if n <= max_points:
        return arr
    step = max(1, n // max_points)
    return arr[::step]


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def mean_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and not math.isnan(v)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ------------------------------------------------------------
# Strava-specific helpers
# ------------------------------------------------------------
def get_activity(activity_id: int, token: str) -> Dict[str, Any]:
    """
    Full activity detail from Strava.
    """
    return strava_get(f"/activities/{activity_id}", token)


def get_recent_activities(per_page: int, token: str) -> List[Dict[str, Any]]:
    per_page = clamp_int(per_page, 1, 50)
    return strava_get("/athlete/activities", token, params={"per_page": per_page})


def get_activity_streams(
    activity_id: int,
    token: str,
    keys: List[str],
    resolution: str = "low",
    series_type: str = "time",
) -> Dict[str, Any]:
    """
    Strava streams endpoint.

    resolution: low|medium|high (low helps keep payload smaller)
    series_type: time|distance
    """
    params = {
        "keys": ",".join(keys),
        "key_by_type": "true",
        "resolution": resolution,
        "series_type": series_type,
    }
    return strava_get(f"/activities/{activity_id}/streams", token, params=params)


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/recent-activities")
def recent_activities(per_page: int = 10) -> List[Dict[str, Any]]:
    token = get_access_token()
    activities = get_recent_activities(per_page, token)

    # lightweight summary list
    out: List[Dict[str, Any]] = []
    for a in activities:
        out.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "type": a.get("type"),
                "start_date": a.get("start_date"),
                "distance_miles": round(meters_to_miles(a.get("distance", 0.0) or 0.0), 2),
                "moving_time": seconds_to_hhmmss(int(a.get("moving_time", 0) or 0)),
                "elapsed_time": seconds_to_hhmmss(int(a.get("elapsed_time", 0) or 0)),
                "elevation_gain_ft": round(meters_to_feet(a.get("total_elevation_gain", 0.0) or 0.0), 0),
                "avg_hr": a.get("average_heartrate"),
                "max_hr": a.get("max_heartrate"),
                "avg_power_w": a.get("average_watts"),
                "weighted_power_w": a.get("weighted_average_watts"),
            }
        )
    return out


# -------------------------
# NEW endpoint #1: /activity
# -------------------------
@app.get("/activity")
def activity(activity_id: int) -> Dict[str, Any]:
    token = get_access_token()
    a = get_activity(activity_id, token)

    # A curated but detailed response (keeps payload sane)
    # You can add fields later if needed.
    return {
        "id": a.get("id"),
        "name": a.get("name"),
        "type": a.get("type"),
        "start_date": a.get("start_date"),
        "timezone": a.get("timezone"),
        "trainer": a.get("trainer"),
        "distance_miles": round(meters_to_miles(a.get("distance", 0.0) or 0.0), 3),
        "moving_time_sec": a.get("moving_time"),
        "moving_time": seconds_to_hhmmss(int(a.get("moving_time", 0) or 0)),
        "elapsed_time": seconds_to_hhmmss(int(a.get("elapsed_time", 0) or 0)),
        "elevation_gain_ft": round(meters_to_feet(a.get("total_elevation_gain", 0.0) or 0.0), 0),
        "avg_hr": a.get("average_heartrate"),
        "max_hr": a.get("max_heartrate"),
        "avg_power_w": a.get("average_watts"),
        "weighted_power_w": a.get("weighted_average_watts"),
        "avg_speed_mph": round((a.get("average_speed", 0.0) or 0.0) * 2.236936, 2),
        "description": a.get("description"),
        # keep raw minimal but useful:
        "splits_standard_available": bool(a.get("splits_standard")),
        "has_heartrate": bool(a.get("has_heartrate")),
        "device_name": (a.get("device_name") or None),
        "gear_id": (a.get("gear_id") or None),
    }


# -------------------------
# Run: mile-by-mile splits
# -------------------------
@app.get("/run-splits")
def run_splits(activity_id: int) -> Dict[str, Any]:
    """
    Returns mile-by-mile splits using Strava's splits_standard.
    This includes pace + elevation_difference + avg HR per mile (when available).
    """
    token = get_access_token()
    a = get_activity(activity_id, token)

    if (a.get("type") or "").lower() not in ["run", "trailrun", "virtualrun"]:
        # allow anyway, but warn
        activity_type = a.get("type")
    else:
        activity_type = a.get("type")

    splits = a.get("splits_standard") or []
    if not splits:
        raise HTTPException(
            status_code=400,
            detail="No splits_standard found on this activity. (Strava may not provide splits for this activity.)",
        )

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(splits, start=1):
        # Strava splits_standard typically provides: distance (m), moving_time, elapsed_time, average_speed, elevation_difference, average_heartrate
        avg_speed = safe_float(s.get("average_speed"))
        pace = speed_mps_to_min_per_mile(avg_speed) if avg_speed else None

        elev_delta_m = safe_float(s.get("elevation_difference"))
        elev_delta_ft = round(meters_to_feet(elev_delta_m), 0) if elev_delta_m is not None else None

        rows.append(
            {
                "mile": i,
                "pace_min_per_mile": pace,
                "elevation_delta_ft": elev_delta_ft,
                "avg_hr": s.get("average_heartrate"),
                "moving_time": seconds_to_hhmmss(int(s.get("moving_time", 0) or 0)),
                "distance_miles": round(meters_to_miles(s.get("distance", 0.0) or 0.0), 3),
            }
        )

    return {
        "activity_id": a.get("id"),
        "name": a.get("name"),
        "type": activity_type,
        "start_date": a.get("start_date"),
        "distance_miles": round(meters_to_miles(a.get("distance", 0.0) or 0.0), 3),
        "moving_time": seconds_to_hhmmss(int(a.get("moving_time", 0) or 0)),
        "avg_hr": a.get("average_heartrate"),
        "max_hr": a.get("max_heartrate"),
        "splits": rows,
        "notes": "Using Strava splits_standard (mile splits). Pace is formatted as min/mi; elevation is delta per mile.",
    }


# ------------------------------------------------------------
# NEW endpoint #2: Bike blocks (watts+HR+elevation, aggregated)
# ------------------------------------------------------------
@app.get("/bike-blocks")
def bike_blocks(
    activity_id: int,
    block_sec: int = 60,
    merge_tolerance_w: int = 10,
    min_block_min: int = 3,
    resolution: str = "low",
) -> Dict[str, Any]:
    """
    Returns time-bucketed blocks that are merged into stable segments.
    This avoids returning second-by-second streams and prevents ResponseTooLargeError in GPT tool calls.

    Parameters:
      - block_sec: bucket size in seconds (default 60)
      - merge_tolerance_w: merge adjacent buckets if avg watts within this tolerance (default 10W)
      - min_block_min: discard blocks shorter than this (default 3 min)
      - resolution: Strava stream resolution low|medium|high (default low)
    """
    token = get_access_token()
    a = get_activity(activity_id, token)

    # Pull streams at low resolution to keep payload reasonable.
    # We do NOT return the raw stream; we compute blocks and return small output.
    keys = ["time", "watts", "heartrate", "altitude", "distance", "cadence"]
    streams = get_activity_streams(activity_id, token, keys=keys, resolution=resolution, series_type="time")

    time_s = streams.get("time", {}).get("data") or []
    watts = streams.get("watts", {}).get("data") or []
    hr = streams.get("heartrate", {}).get("data") or []
    alt = streams.get("altitude", {}).get("data") or []
    dist = streams.get("distance", {}).get("data") or []

    if not time_s or not watts:
        raise HTTPException(
            status_code=400,
            detail="No time/watts stream available for this activity. (Power data may not be present on Strava for this ride.)",
        )

    block_sec = clamp_int(block_sec, 10, 600)
    merge_tolerance_w = clamp_int(merge_tolerance_w, 1, 50)
    min_block_min = clamp_int(min_block_min, 1, 30)
    min_block_sec = min_block_min * 60

    n = len(time_s)

    def get_at(arr: List[Any], i: int) -> Optional[float]:
        if not arr or i < 0 or i >= len(arr):
            return None
        return safe_float(arr[i])

    # 1) Bucket into fixed time windows
    buckets: List[Dict[str, Any]] = []
    start_idx = 0

    while start_idx < n:
        start_t = int(time_s[start_idx] or 0)
        end_t = start_t + block_sec

        # advance end_idx
        end_idx = start_idx
        while end_idx < n and int(time_s[end_idx] or 0) < end_t:
            end_idx += 1

        if end_idx <= start_idx:
            end_idx = start_idx + 1

        w_vals = [safe_float(watts[i]) for i in range(start_idx, min(end_idx, len(watts)))]
        h_vals = [get_at(hr, i) for i in range(start_idx, end_idx)]
        a0 = get_at(alt, start_idx)
        a1 = get_at(alt, end_idx - 1)
        d0 = get_at(dist, start_idx)
        d1 = get_at(dist, end_idx - 1)

        avg_w = mean_ignore_none(w_vals)
        avg_h = mean_ignore_none(h_vals)

        buckets.append(
            {
                "start_sec": start_t,
                "end_sec": int(time_s[end_idx - 1] or end_t),
                "duration_sec": int(time_s[end_idx - 1] or end_t) - start_t,
                "avg_watts": avg_w,
                "avg_hr": avg_h,
                "alt_delta_ft": round(meters_to_feet((a1 - a0) if (a0 is not None and a1 is not None) else 0.0), 0)
                if (a0 is not None and a1 is not None)
                else None,
                "dist_miles": round(meters_to_miles((d1 - d0) if (d0 is not None and d1 is not None) else 0.0), 3)
                if (d0 is not None and d1 is not None)
                else None,
            }
        )

        start_idx = end_idx

    # 2) Merge adjacent buckets when watts are "similar"
    merged: List[Dict[str, Any]] = []
    for b in buckets:
        if b["avg_watts"] is None:
            continue

        if not merged:
            merged.append(b)
            continue

        prev = merged[-1]
        if prev["avg_watts"] is None:
            merged[-1] = b
            continue

        if abs(float(b["avg_watts"]) - float(prev["avg_watts"])) <= merge_tolerance_w:
            # merge into prev
            total_dur = (prev["duration_sec"] or 0) + (b["duration_sec"] or 0)
            if total_dur <= 0:
                continue

            # weighted avg watts/hr by duration
            prev_w = float(prev["avg_watts"])
            b_w = float(b["avg_watts"])
            prev_d = float(prev["duration_sec"] or 0)
            b_d = float(b["duration_sec"] or 0)

            prev["end_sec"] = b["end_sec"]
            prev["duration_sec"] = int(total_dur)
            prev["avg_watts"] = (prev_w * prev_d + b_w * b_d) / total_dur

            # HR
            if prev.get("avg_hr") is not None and b.get("avg_hr") is not None:
                prev_h = float(prev["avg_hr"])
                b_h = float(b["avg_hr"])
                prev["avg_hr"] = (prev_h * prev_d + b_h * b_d) / total_dur
            elif prev.get("avg_hr") is None:
                prev["avg_hr"] = b.get("avg_hr")

            # altitude delta, distance
            if prev.get("alt_delta_ft") is not None and b.get("alt_delta_ft") is not None:
                prev["alt_delta_ft"] = round(float(prev["alt_delta_ft"]) + float(b["alt_delta_ft"]), 0)
            if prev.get("dist_miles") is not None and b.get("dist_miles") is not None:
                prev["dist_miles"] = round(float(prev["dist_miles"]) + float(b["dist_miles"]), 3)

        else:
            merged.append(b)

    # 3) Filter short blocks + format nicely
    blocks: List[Dict[str, Any]] = []
    for m in merged:
        dur = int(m.get("duration_sec") or 0)
        if dur < min_block_sec:
            continue

        blocks.append(
            {
                "start_min": round((m["start_sec"] or 0) / 60.0, 1),
                "end_min": round((m["end_sec"] or 0) / 60.0, 1),
                "duration_min": round(dur / 60.0, 1),
                "avg_watts": int(round(float(m["avg_watts"]))),
                "avg_hr": int(round(float(m["avg_hr"]))) if m.get("avg_hr") is not None else None,
                "alt_delta_ft": m.get("alt_delta_ft"),
                "dist_miles": m.get("dist_miles"),
            }
        )

    return {
        "activity_id": a.get("id"),
        "name": a.get("name"),
        "type": a.get("type"),
        "start_date": a.get("start_date"),
        "distance_miles": round(meters_to_miles(a.get("distance", 0.0) or 0.0), 3),
        "moving_time": seconds_to_hhmmss(int(a.get("moving_time", 0) or 0)),
        "avg_power_w": a.get("average_watts"),
        "weighted_power_w": a.get("weighted_average_watts"),
        "avg_hr": a.get("average_heartrate"),
        "max_hr": a.get("max_heartrate"),
        "block_settings": {
            "block_sec": block_sec,
            "merge_tolerance_w": merge_tolerance_w,
            "min_block_min": min_block_min,
            "resolution": resolution,
        },
        "blocks": blocks,
        "notes": "Blocks are time-bucketed and merged by similar watts so you can analyze intervals without second-by-second data.",
    }


# ------------------------------------------------------------
# Optional debug endpoint (downsampled streams)
# ------------------------------------------------------------
@app.get("/bike-streams")
def bike_streams(activity_id: int, max_points: int = 600) -> Dict[str, Any]:
    """
    Debug endpoint: returns downsampled stream arrays.
    This is NOT recommended for GPT analysis; use /bike-blocks instead.
    """
    token = get_access_token()
    keys = ["time", "watts", "heartrate", "altitude", "distance", "cadence"]
    streams = get_activity_streams(activity_id, token, keys=keys, resolution="low", series_type="time")

    def ds(key: str) -> List[Any]:
        arr = streams.get(key, {}).get("data") or []
        return downsample_series(arr, max_points=max_points)

    return {
        "activity_id": activity_id,
        "max_points": max_points,
        "time_s": ds("time"),
        "watts": ds("watts"),
        "heartrate": ds("heartrate"),
        "altitude_m": ds("altitude"),
        "distance_m": ds("distance"),
        "cadence_rpm": ds("cadence"),
    }
