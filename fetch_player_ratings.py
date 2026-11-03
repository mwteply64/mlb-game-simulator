"""Download per-player defense and baserunning ratings.

Two Statcast sources, one per season, cached as parquet:

* **Outs Above Average** -- range-based fielding, with `fielding_runs_prevented`
  already expressed in runs. Aggregated later across whoever is actually on the
  field for a given plate appearance.
* **Sprint speed** -- feet per second on competitive runs.

Sprint speed rather than FanGraphs BsR/UBR is a deliberate choice, not just a
workaround for FanGraphs blocking the legacy endpoint. BsR measures *runs added
by baserunning*, which is precisely what the empirical base-out transition
tables already measure; conditioning those tables on BsR would be circular.
Sprint speed is an independent physical measurement of the thing that causes
the advancement, so it can condition the transitions without feeding their own
output back in.

Known gaps this leaves, both recorded rather than silently ignored:
* **No DRS.** FanGraphs returns 403. OAA covers range; DRS additionally covers
  arm strength and some positioning. OAA is the modern standard and is
  sufficient, but the arm component is genuinely missing.
* **No catcher framing.** Savant's framing endpoint changed format and the
  parser fails. Framing is worth roughly one to two wins a season for the
  extreme catchers and acts on called strikes, so it belongs in the strikeout
  and walk channel rather than the balls-in-play channel. Left for later.

    python fetch_player_ratings.py
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import pybaseball as pb

import config

OAA_PATH = os.path.join(config.RAW_DIR, "oaa_by_season.parquet")
SPEED_PATH = os.path.join(config.RAW_DIR, "sprint_speed_by_season.parquet")


def fetch_oaa(seasons):
    frames = []
    for year in seasons:
        try:
            df = pb.statcast_outs_above_average(year, pos="all", min_att=1)
        except Exception as exc:  # noqa: BLE001
            print(f"  {year}: FAILED {exc}")
            continue
        df["season"] = year
        keep = ["season", "player_id", "primary_pos_formatted",
                "outs_above_average", "fielding_runs_prevented",
                "outs_above_average_rhh", "outs_above_average_lhh"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])
        print(f"  {year}: {len(df):,} fielders")
        time.sleep(1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_speed(seasons):
    frames = []
    for year in seasons:
        try:
            df = pb.statcast_sprint_speed(year, min_opp=1)
        except Exception as exc:  # noqa: BLE001
            print(f"  {year}: FAILED {exc}")
            continue
        df["season"] = year
        keep = ["season", "player_id", "position", "age", "competitive_runs",
                "sprint_speed", "hp_to_1b"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])
        print(f"  {year}: {len(df):,} runners")
        time.sleep(1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)

    print("outs above average:")
    oaa = fetch_oaa(config.SEASONS)
    if len(oaa):
        oaa.to_parquet(OAA_PATH, index=False)
        print(f"  wrote {len(oaa):,} rows -> {OAA_PATH}")

    print("\nsprint speed:")
    speed = fetch_speed(config.SEASONS)
    if len(speed):
        speed.to_parquet(SPEED_PATH, index=False)
        print(f"  wrote {len(speed):,} rows -> {SPEED_PATH}")
        print(f"\n  league sprint speed by season:")
        print(speed.groupby("season")["sprint_speed"].agg(
            ["mean", "std", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
