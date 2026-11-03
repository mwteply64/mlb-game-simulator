"""Historical weather for every game, so weather effects can be fitted.

Yahoo hands us a forecast for tonight, which is enough to *apply* a weather
adjustment but not to *fit* one -- there is no history in the plate-appearance
table to test against. Everything else in this project carries a measurement;
an imported weather coefficient would be the only asserted number in it, in a
project where roughly half the plausible-sounding adjustments have failed their
controls.

Open-Meteo's archive is free and needs no key, and takes a date range, so this
costs one request per park per season rather than one per game -- about 180
calls for 2021-2026 instead of fifteen thousand.

**Wind direction is stored raw, in degrees, and deliberately not converted to
"blowing out".** That conversion needs each park's home-plate-to-centre-field
bearing, which no API publishes. Storing sin and cos of the direction lets a
per-park fit discover the orientation from the run-scoring data itself, which is
both more honest and less work than sourcing 30 stadium bearings.

    python fetch_weather.py
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
STATSAPI = "https://statsapi.mlb.com/api/v1"
PATH = os.path.join(config.RAW_DIR, "game_weather.parquet")

ABBREV = {"AZ": "AZ", "ARI": "AZ", "CWS": "CWS", "CHW": "CWS", "OAK": "ATH",
          "ATH": "ATH"}


def park_coords():
    """Home park latitude and longitude, by the abbreviation the project uses."""
    teams = requests.get(f"{STATSAPI}/teams", params={"sportId": 1},
                         timeout=30).json().get("teams", [])
    venues = requests.get(f"{STATSAPI}/venues",
                          params={"sportId": 1, "hydrate": "location"},
                          timeout=30).json().get("venues", [])
    coord = {}
    for v in venues:
        c = (v.get("location", {}) or {}).get("defaultCoordinates") or {}
        if c.get("latitude") is not None:
            coord[v["id"]] = (float(c["latitude"]), float(c["longitude"]))

    out = {}
    for t in teams:
        ab = (t.get("abbreviation") or "").upper()
        ab = ABBREV.get(ab, ab)
        vid = (t.get("venue") or {}).get("id")
        if vid in coord:
            out[ab] = coord[vid]
    return out


def game_times(season):
    """Game start times, so weather is read at first pitch rather than midnight."""
    # `hydrate=team` is required: without it the schedule returns team ids and
    # names but no abbreviation, so every park keyed to an empty string and
    # nothing matched.
    r = requests.get(f"{STATSAPI}/schedule", params={
        "sportId": 1, "startDate": f"{season}-03-01",
        "endDate": f"{season}-11-15", "hydrate": "team"}, timeout=90)
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            home = (g["teams"]["home"]["team"].get("abbreviation") or "").upper()
            rows.append({"game_pk": int(g["gamePk"]),
                         "start_utc": pd.to_datetime(g["gameDate"], utc=True),
                         "park": ABBREV.get(home, home)})
    return pd.DataFrame(rows)


def fetch_park_season(lat, lon, season, pause=0.4):
    # The archive rejects a future end date, which silently failed every 2026
    # park until the range was clamped -- the season runs past today.
    import datetime as _dt
    end = min(_dt.date(season, 11, 15),
              _dt.date.today() - _dt.timedelta(days=1))
    if end < _dt.date(season, 3, 1):
        return None
    r = requests.get(ARCHIVE, params={
        "latitude": lat, "longitude": lon,
        "start_date": f"{season}-03-01", "end_date": end.isoformat(),
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,"
                  "precipitation,relative_humidity_2m",
        "timezone": "UTC"}, timeout=90)
    time.sleep(pause)
    if r.status_code != 200:
        return None
    h = r.json().get("hourly") or {}
    if not h.get("time"):
        return None
    d = pd.DataFrame(h)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d


def main():
    coords = park_coords()
    print(f"{len(coords)} parks with coordinates")

    have = set()
    if os.path.exists(PATH):
        old = pd.read_parquet(PATH)
        have = set(zip(old["park"], old["season"]))
        print(f"  {len(old):,} games already stored")
    else:
        old = None

    frames = [old] if old is not None else []
    for season in config.SEASONS:
        sched = game_times(season)
        if not len(sched):
            continue
        print(f"\n{season}: {len(sched):,} scheduled games")
        for park, (lat, lon) in sorted(coords.items()):
            if (park, season) in have:
                continue
            games = sched[sched["park"] == park]
            if not len(games):
                continue
            wx = fetch_park_season(lat, lon, season)
            if wx is None:
                print(f"  {park}: FAILED")
                continue
            # Nearest hour to first pitch.
            g = games.sort_values("start_utc")
            m = pd.merge_asof(g, wx.sort_values("time"), left_on="start_utc",
                              right_on="time", direction="nearest",
                              tolerance=pd.Timedelta("2h"))
            m["season"] = season
            frames.append(m.drop(columns=["time"]))
            print(f"  {park}: {len(m)} games matched")

    if not frames:
        raise SystemExit("nothing fetched")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("game_pk")
    # Direction as components, so a per-park fit can find the orientation.
    wd = np.radians(pd.to_numeric(out["wind_direction_10m"], errors="coerce"))
    out["wind_sin"] = np.sin(wd)
    out["wind_cos"] = np.cos(wd)
    out.to_parquet(PATH, index=False)
    print(f"\nwrote {len(out):,} games -> {PATH}")
    print(f"  temperature   {out['temperature_2m'].mean():.1f}C mean")
    print(f"  wind speed    {out['wind_speed_10m'].mean():.1f} km/h mean")
    print(f"  seasons       {sorted(out['season'].dropna().unique())}")


if __name__ == "__main__":
    main()
