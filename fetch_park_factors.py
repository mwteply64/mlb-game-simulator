"""Statcast park factors from Baseball Savant.

These replace the home-grown factors in `outcome_model.fit_park_factors`, which
compared each park's rates against the *league* -- a construction that conflates
the park with the quality of the teams playing in it. If a club's pitching is
poor, its park looks more hitter-friendly than it is. Savant's are built the
correct way, comparing each team's home performance to its own road
performance, which cancels team quality.

They also arrive as a three-year rolling window with an index per outcome
(100 = neutral), which is the exact shape the simulator wants.

    python fetch_park_factors.py
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import json
import re

import numpy as np
import pandas as pd
import requests

import config

# No custom User-Agent, matching record_odds.py. Savant and StatsAPI both accept
# any value today, so this is hardening rather than a fix: the string this used
# to send, "Mozilla/5.0 (research; personal use)", is a malformed browser UA
# that ESPN began rejecting with a 403 on 2026-08-04 and silently cost a day of
# collection. Sending requests' own identifier is the truthful option and the
# one least likely to trip a spoof heuristic if Savant ever adds the same rule.
UA = None
URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
STATSAPI = "https://statsapi.mlb.com/api/v1"
PATH = os.path.join(config.RAW_DIR, "park_factors.parquet")

# Savant publishes an index per outcome; these are the ones the simulator uses.
INDEX_COLS = {"index_so": "K", "index_bb": "BB", "index_1b": "1B",
              "index_2b": "2B", "index_3b": "3B", "index_hr": "HR"}


def team_abbrev_by_id():
    r = requests.get(f"{STATSAPI}/teams", params={"sportId": 1}, timeout=30,
                     headers=UA)
    out = {}
    for t in r.json().get("teams", []):
        ab = (t.get("abbreviation") or "").upper()
        out[t["id"]] = {"AZ": "AZ", "CWS": "CWS", "OAK": "ATH"}.get(ab, ab)
    return out


def fetch_year(year):
    r = requests.get(URL, params={"type": "year", "year": year}, timeout=40,
                     headers=UA)
    if r.status_code != 200:
        return None
    m = re.search(r"var\s+data\s*=\s*(\[.*?\]);", r.text, re.S)
    if not m:
        return None
    df = pd.DataFrame(json.loads(m.group(1)))
    df["season"] = year
    return df


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    abbrev = team_abbrev_by_id()

    frames = []
    for year in config.SEASONS:
        df = fetch_year(year)
        if df is None or not len(df):
            print(f"  {year}: FAILED")
            continue
        df["park"] = pd.to_numeric(df["main_team_id"],
                                   errors="coerce").astype("Int64").map(abbrev)
        keep = ["season", "park", "venue_name", "n_pa", "year_range"] + \
               list(INDEX_COLS)
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])
        print(f"  {year}: {len(df)} parks  ({df['year_range'].iloc[0]})")
        time.sleep(0.5)

    if not frames:
        raise SystemExit("no park factors retrieved")

    out = pd.concat(frames, ignore_index=True)
    for c in INDEX_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_parquet(PATH, index=False)
    print(f"\nwrote {len(out):,} park-seasons -> {PATH}")

    print("\nmost extreme parks, 2025 (index, 100 = neutral):")
    latest = out[out["season"] == max(out["season"])]
    for col, name in INDEX_COLS.items():
        s = latest.set_index("park")[col].dropna().sort_values()
        if len(s):
            print(f"  {name:<3} low {s.index[0]} {s.iloc[0]:.0f}   "
                  f"high {s.index[-1]} {s.iloc[-1]:.0f}")


def load(season):
    """Per-outcome multipliers by park, as the simulator expects them.

    Uses the row for `season - 1`, not `season`. Savant's window ends at the
    year requested -- the 2024 row covers 2022-2024 -- so taking the same year
    would feed the season being projected back into its own park factors. The
    previous year's row (2021-2023 for a 2024 projection) is the leak-free
    choice, and park effects move slowly enough that the cost is negligible.

    Outcomes Savant does not publish an index for (hit by pitch, outs in play)
    stay neutral; the vector is renormalised downstream, so the outs share
    absorbs whatever the others give up.
    """
    if not os.path.exists(PATH):
        return None
    all_df = pd.read_parquet(PATH)
    df = all_df[all_df["season"] == season - 1]
    if not len(df):
        available = all_df[all_df["season"] < season]
        if not len(available):
            return None
        df = all_df[all_df["season"] == available["season"].max()]

    out = {}
    for row in df.itertuples(index=False):
        v = np.ones(len(config.OUTCOMES))
        for col, name in INDEX_COLS.items():
            val = getattr(row, col, None)
            if val is not None and np.isfinite(val):
                v[config.OUTCOMES.index(name)] = float(val) / 100.0
        out[row.park] = v
    return out


if __name__ == "__main__":
    main()
