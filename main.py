import os
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

if not CLIENT_ID or not CLIENT_SECRET or not REFRESH_TOKEN:
    # Don’t crash at import-time on Render; just make it obvious in the endpoint errors.
    pass


def get_access_token() -> str:
    r = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        timeout=20,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Token refresh failed: {r.text}")
    return r.json()["access_token"]


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/recent-activities")
def recent_activities(per_page: int = 10):
    token = get_access_token()
    r = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        params={"per_page": per_page},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Strava fetch failed: {r.text}")

    activities = r.json()

    return [
        {
            "id": a["id"],
            "name": a.get("name"),
            "type": a.get("type"),
            "start_date": a.get("start_date"),
            "distance_km": (a.get("distance") or 0) / 1000,
            "moving_time_min": (a.get("moving_time") or 0) / 60,
            "elapsed_time_min": (a.get("elapsed_time") or 0) / 60,
            "elevation_m": a.get("total_elevation_gain"),
            "avg_hr": a.get("average_heartrate"),
            "max_hr": a.get("max_heartrate"),
            "avg_power": a.get("average_watts"),
            "weighted_power": a.get("weighted_average_watts"),
        }
        for a in activities
    ]
