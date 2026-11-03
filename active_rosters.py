"""Who was actually on the active roster, by team and date.

The simulator inferred availability from recent appearances: a reliever counted
as available if he had pitched in the last 21 days. That cannot distinguish an
injured pitcher from a rested one, so an arm placed on the IL kept being
simulated for up to three weeks. Measured on 2026-08-04, 62 pitchers who had not
appeared in 10+ days still carried 8.7% of league bullpen usage weight, and 42
of them had been idle 14+ days. Every one of those absorbed relief innings that
belonged to a pitcher who could actually take the ball.

StatsAPI answers this directly. `rosterType=active` returns only status "A" --
the 40-man view of the same team shows what is being excluded: D10, D15 and D60
injured designations plus minor-league reassignments.

Crucially the endpoint honours a `date` parameter, so this is not just a live
fix. Historical rosters are available (a team's June 1 and August 4 active
rosters shared 15 of 26 players), which means backtests get the same correction
rather than being silently more optimistic than live.

**Positions are not used for classification.** Every pitcher comes back labelled
simply "P" -- the feed does not separate starters from relievers, and it should
not be asked to. Roles are decided by behaviour in `bullpen.relief_appearances`,
which reads how a pitcher was actually used. This file only ever answers "was he
on the roster", never "what is he".

    python active_rosters.py                     # today, all 30 clubs
    python active_rosters.py --backfill 2026     # weekly snapshots for a season
"""

import argparse
import datetime as dt
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config

STATSAPI = "https://statsapi.mlb.com/api/v1"
PATH = os.path.join(config.RAW_DIR, "active_rosters.parquet")

# Snapshot cadence for backfills. Rosters move constantly, but an IL stint lasts
# at minimum 10 days, so weekly sampling catches essentially every stint while
# costing a thirtieth of the requests a daily pull would.
DEFAULT_EVERY = 7


def team_abbrev():
    """StatsAPI team id -> the abbreviation used everywhere else in the project."""
    r = requests.get(f"{STATSAPI}/teams", params={"sportId": 1}, timeout=30)
    out = {}
    for t in r.json().get("teams", []):
        ab = (t.get("abbreviation") or "").upper()
        out[int(t["id"])] = {"AZ": "AZ", "CWS": "CWS", "OAK": "ATH"}.get(ab, ab)
    return out


def fetch_date(date_str, abbrev, pause=0.12):
    rows = []
    for tid, ab in abbrev.items():
        try:
            r = requests.get(f"{STATSAPI}/teams/{tid}/roster",
                             params={"rosterType": "active", "date": date_str},
                             timeout=30)
            if r.status_code != 200:
                print(f"  {date_str} {ab}: HTTP {r.status_code}")
                continue
            for p in r.json().get("roster", []):
                pos = p.get("position", {}) or {}
                rows.append({
                    "date": date_str,
                    "team": ab,
                    "player_id": int(p["person"]["id"]),
                    "position": pos.get("abbreviation"),
                    "is_pitcher": pos.get("type") == "Pitcher",
                })
        except requests.RequestException as exc:
            print(f"  {date_str} {ab}: {type(exc).__name__}")
        time.sleep(pause)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int,
                    help="season to backfill weekly snapshots for")
    ap.add_argument("--every", type=int, default=DEFAULT_EVERY)
    args = ap.parse_args()

    os.makedirs(config.RAW_DIR, exist_ok=True)
    abbrev = team_abbrev()
    print(f"{len(abbrev)} clubs")

    dates = []
    if args.backfill:
        start = dt.date(args.backfill, 3, 20)
        end = min(dt.date(args.backfill, 11, 5), dt.date.today())
        d = start
        while d <= end:
            dates.append(d.isoformat())
            d += dt.timedelta(days=args.every)
    else:
        dates = [dt.date.today().isoformat()]

    existing = pd.read_parquet(PATH) if os.path.exists(PATH) else None
    have = set(existing["date"].unique()) if existing is not None else set()

    frames = [existing] if existing is not None else []
    for ds in dates:
        if ds in have:
            print(f"  {ds}: already stored, skipping")
            continue
        rows = fetch_date(ds, abbrev)
        if rows:
            frames.append(pd.DataFrame(rows))
            print(f"  {ds}: {len(rows)} roster spots")

    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "team", "player_id"])
    out.to_parquet(PATH, index=False)
    print(f"\nwrote {len(out):,} roster-spots across "
          f"{out['date'].nunique()} dates -> {PATH}")


class ActiveRosters:
    """Active-roster membership, with the nearest prior snapshot standing in.

    Returns None from `as_of` when there is no snapshot at or before the date
    asked for, and every caller treats None as "no information" rather than "no
    players". A missing roster file must leave behaviour exactly as it was --
    silently emptying bullpens would be far worse than the staleness this is
    meant to fix.
    """

    def __init__(self, df=None):
        if df is None:
            df = pd.read_parquet(PATH) if os.path.exists(PATH) else None
        self.df = df
        if df is not None and len(df):
            self.df = df.copy()
            self.df["date"] = pd.to_datetime(self.df["date"])
            self._dates = np.sort(self.df["date"].unique())
            self._by = {k: set(g["player_id"].astype(int))
                        for k, g in self.df.groupby(["team", "date"])}
        else:
            self._dates = np.array([], dtype="datetime64[ns]")
            self._by = {}

    def available(self):
        return len(self._dates) > 0

    def as_of(self, team, date):
        """Set of active player ids, or None if this date predates every snapshot."""
        if not self.available():
            return None
        d = np.datetime64(pd.Timestamp(date).normalize())
        prior = self._dates[self._dates <= d]
        if not len(prior):
            return None
        return self._by.get((team, pd.Timestamp(prior[-1])))


def ensure_today(max_age_days=1, verbose=True):
    """Fetch today's rosters if the newest snapshot is stale, then load.

    Called at the top of the daily props run so pricing never uses yesterday's
    roster. IL moves are announced the morning of a game -- James Wood was
    placed on the IL the same day his team played -- so a snapshot even one day
    old can have a player who is now unavailable batting in a simulated lineup.

    Network failure is not fatal: it falls back to whatever is already stored
    and says how old that is, because pricing on a slightly stale roster beats
    not pricing at all. Silence would be the real error.
    """
    today = dt.date.today()
    have = None
    if os.path.exists(PATH):
        try:
            have = pd.to_datetime(pd.read_parquet(PATH)["date"]).max().date()
        except Exception:
            have = None

    if have is not None and (today - have).days <= max_age_days:
        if verbose:
            print(f"  rosters current ({have})")
        return ActiveRosters()

    if verbose:
        print(f"  refreshing rosters (newest snapshot: {have or 'none'})...")
    try:
        abbrev = team_abbrev()
        rows = fetch_date(today.isoformat(), abbrev)
        if rows:
            frames = [pd.read_parquet(PATH)] if os.path.exists(PATH) else []
            frames.append(pd.DataFrame(rows))
            out = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["date", "team", "player_id"])
            out.to_parquet(PATH, index=False)
            if verbose:
                print(f"  rosters updated: {len(rows)} spots for {today}")
        elif verbose:
            print("  WARNING: roster fetch returned nothing; using stored")
    except Exception as exc:
        if verbose:
            age = f"{(today - have).days}d old" if have else "none stored"
            print(f"  WARNING: roster refresh failed ({type(exc).__name__}); "
                  f"using stored ({age})")

    return ActiveRosters()


def load():
    return ActiveRosters()


if __name__ == "__main__":
    main()
