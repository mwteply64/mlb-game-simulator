"""Record MLB betting lines prospectively, for an un-overfittable benchmark.

The free historical archive stops at 2021, so the market comparison cannot be
backtested against any season this project can model. Recording lines forward
from today solves that, and solves it better: a prospective test cannot be
overfit, because the predictions are made before the outcomes exist.

Two keyless sources:

* **MLB StatsAPI** for the schedule -- `gamePk`, first-pitch time, and teams.
  `gamePk` is the same identifier the Statcast data uses, so the join is exact
  and none of the date/team/doubleheader matching that the historical archive
  needed applies here.
* **ESPN's core API** for the lines -- moneyline, total, and runline.

Snapshots are appended with a timestamp rather than overwritten. The closing
line is then whatever the last snapshot before first pitch happened to be,
which is the only way to capture it when games start at different times all
afternoon and evening. Run this several times a day; more snapshots means a
tighter approximation of the true close.

    python record_odds.py              # snapshot today and tomorrow
    python record_odds.py --days 3
    python record_odds.py --closing    # derive closing lines from the log
"""

import argparse
import datetime as dt
import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import requests

import config

# No custom User-Agent. This used to send "Mozilla/5.0 (research; personal
# use)", which ESPN began rejecting with a 403 on 2026-08-04 and silently
# stopped a day of collection. Testing showed ESPN accepts requests' own default
# (python-requests/2.x) and 403s *every* custom string, honest ones included --
# the old value looked like a malformed browser UA and tripped a spoof check.
# Sending the library's own identifier is both the truthful option and the one
# that works: it says plainly that this is a script.
UA = None
STATSAPI = "https://statsapi.mlb.com/api/v1"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"

ODDS_LOG = os.path.join(config.PROCESSED_DIR, "odds_snapshots.parquet")
CLOSING = os.path.join(config.PROCESSED_DIR, "odds_closing.parquet")
PROPS_LOG = os.path.join(config.PROCESSED_DIR, "prop_snapshots.parquet")
ATHLETE_CACHE = os.path.join(config.PROCESSED_DIR, "espn_athletes.parquet")

ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"

# Props are only fetched for games starting within this window. Each game costs
# six paged requests against an undocumented endpoint, so there is no reason to
# poll a 10pm start at breakfast -- and the lines that matter are the late ones
# anyway.
PROP_WINDOW_HOURS = 8

# StatsAPI abbreviations differ from Statcast's for a few clubs.
ABBREV_FIXES = {"AZ": "AZ", "ARI": "AZ", "CWS": "CWS", "CHW": "CWS",
                "WSH": "WSH", "WSN": "WSH", "SD": "SD", "SDP": "SD",
                "SF": "SF", "SFG": "SF", "TB": "TB", "TBR": "TB",
                "KC": "KC", "KCR": "KC", "OAK": "ATH", "ATH": "ATH"}


def get_json(url, params=None, tries=3):
    """Fetch JSON, and make a persistent HTTP failure visible.

    The silent version of this cost a full day of collection: ESPN started
    returning 403, `get_json` returned None, and the caller reported "0 matched
    to ESPN" -- indistinguishable from a quiet day with no games. The scheduled
    task exited 0 throughout. Anything that fails every attempt now says so.
    """
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers=UA)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    print(f"  FETCH FAILED ({last}): {url}")
    return None


def team_abbrev_map():
    j = get_json(f"{STATSAPI}/teams", {"sportId": 1})
    out = {}
    if j:
        for t in j.get("teams", []):
            ab = (t.get("abbreviation") or "").upper()
            out[t["id"]] = ABBREV_FIXES.get(ab, ab)
    return out


def schedule(date_str, abbrev):
    """Games on a date, with gamePk, first-pitch time, and team codes."""
    j = get_json(f"{STATSAPI}/schedule", {"sportId": 1, "date": date_str})
    games = []
    for d in (j or {}).get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            games.append({
                "game_pk": g["gamePk"],
                "game_date": d["date"],
                "start_utc": pd.to_datetime(g["gameDate"], utc=True),
                "home_team": abbrev.get(home["id"], ""),
                "away_team": abbrev.get(away["id"], ""),
                "home_name": home.get("name", ""),
                "away_name": away.get("name", ""),
                "status": g.get("status", {}).get("abstractGameState", ""),
                "double_header": g.get("doubleHeader", "N"),
                "game_number": g.get("gameNumber", 1),
            })
    return games


def espn_events(date_str):
    """ESPN events for a date, keyed for matching against the schedule."""
    j = get_json(f"{ESPN_SITE}/scoreboard",
                 {"dates": date_str.replace("-", "")})
    out = []
    for e in (j or {}).get("events", []):
        comps = e.get("competitions", [{}])
        comp = comps[0] if comps else {}
        names = {}
        for c in comp.get("competitors", []):
            names[c.get("homeAway")] = c.get("team", {}).get("displayName", "")
        out.append({
            "event_id": e.get("id"),
            "competition_id": comp.get("id", e.get("id")),
            "start_utc": pd.to_datetime(e.get("date"), utc=True, errors="coerce"),
            "home_name": names.get("home", ""),
            "away_name": names.get("away", ""),
        })
    return out


def event_odds(event_id, competition_id):
    j = get_json(f"{ESPN_CORE}/events/{event_id}/competitions/{competition_id}/odds")
    rows = []
    for item in (j or {}).get("items", []):
        home = item.get("homeTeamOdds", {}) or {}
        away = item.get("awayTeamOdds", {}) or {}
        rows.append({
            "provider": (item.get("provider", {}) or {}).get("name"),
            "ml_home": home.get("moneyLine"),
            "ml_away": away.get("moneyLine"),
            "total": item.get("overUnder"),
            "over_odds": item.get("overOdds"),
            "under_odds": item.get("underOdds"),
            "runline": item.get("spread"),
        })
    return rows


def match_events(games, events):
    """Pair StatsAPI games to ESPN events by teams, then nearest start time.

    Matching on start time within the team pair is what keeps doubleheaders
    straight without relying on either source's ordering.
    """
    pairs = {}
    for g in games:
        cands = [e for e in events
                 if e["home_name"] == g["home_name"]
                 and e["away_name"] == g["away_name"]]
        if not cands:
            continue
        best = min(cands, key=lambda e: abs(
            (e["start_utc"] - g["start_utc"]).total_seconds())
            if pd.notna(e["start_utc"]) else 1e12)
        pairs[g["game_pk"]] = best
    return pairs


def snapshot(days=2):
    abbrev = team_abbrev_map()
    now = pd.Timestamp.utcnow()
    rows = []

    for offset in range(days):
        date_str = (dt.date.today() + dt.timedelta(days=offset)).isoformat()
        games = schedule(date_str, abbrev)
        if not games:
            print(f"  {date_str}: no games")
            continue
        events = espn_events(date_str)
        pairs = match_events(games, events)
        print(f"  {date_str}: {len(games)} games, {len(pairs)} matched to ESPN")
        # Scheduled games with no ESPN counterpart is not a normal state -- it
        # means the feed is broken or the identifiers moved, not that there is
        # nothing to record. Say so, because the task's exit code will not.
        if games and not pairs:
            print(f"  WARNING: {len(games)} scheduled games but ESPN returned "
                  f"{len(events)} events and none matched -- feed or "
                  f"identifier change, not an empty slate")

        for g in games:
            ev = pairs.get(g["game_pk"])
            if ev is None:
                continue
            for o in event_odds(ev["event_id"], ev["competition_id"]):
                rows.append({**{k: g[k] for k in
                                ("game_pk", "game_date", "start_utc",
                                 "home_team", "away_team", "status",
                                 "game_number")},
                             "snapshot_utc": now, **o})
            time.sleep(0.4)   # be polite to an undocumented endpoint

    if not rows:
        print("no odds captured")
        return None

    new = pd.DataFrame(rows)
    if os.path.exists(ODDS_LOG):
        old = pd.read_parquet(ODDS_LOG)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new

    combined = combined.drop_duplicates(
        subset=["game_pk", "snapshot_utc", "provider"], keep="last")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    combined.to_parquet(ODDS_LOG, index=False)
    print(f"captured {len(new):,} rows; log now {len(combined):,} "
          f"across {combined['game_pk'].nunique():,} games -> {ODDS_LOG}")
    return combined


def _load_athlete_cache():
    if os.path.exists(ATHLETE_CACHE):
        df = pd.read_parquet(ATHLETE_CACHE)
        return dict(zip(df["espn_id"].astype(int), df["name"]))
    return {}


def _save_athlete_cache(cache):
    if not cache:
        return
    pd.DataFrame({"espn_id": list(cache.keys()),
                  "name": list(cache.values())}).to_parquet(
        ATHLETE_CACHE, index=False)


def fetch_props(event_id, cache):
    """Every player prop DraftKings is posting for one game.

    Athlete references are resolved to names and cached across runs -- the same
    few hundred players recur all season, so this settles into almost no
    requests after the first slate. Names, not ESPN ids, are what join back to
    the simulator's MLBAM ids.
    """
    rows, page = [], 1
    while True:
        j = get_json(f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}"
                     f"/odds/100/propBets",
                     {"limit": 100, "page": page})
        if not j:
            break
        for it in j.get("items", []):
            ath_ref = (it.get("athlete") or {}).get("$ref", "")
            espn_id = None
            if "/athletes/" in ath_ref:
                try:
                    espn_id = int(ath_ref.split("/athletes/")[1].split("?")[0])
                except (ValueError, IndexError):
                    espn_id = None
            odds = it.get("odds", {}) or {}
            rows.append({
                "event_id": event_id,
                "espn_athlete_id": espn_id,
                "prop_type": (it.get("type") or {}).get("name"),
                "line": _num((odds.get("total") or {}).get("value")),
                "american": _num((odds.get("american") or {}).get("value")),
                "american_open": _num((odds.get("american") or {}).get("open")),
                "last_updated": it.get("lastUpdated"),
            })
            if espn_id and espn_id not in cache:
                cache[espn_id] = None      # resolved below
        if page >= int(j.get("pageCount", 1)):
            break
        page += 1
        time.sleep(0.25)

    unresolved = [k for k, v in cache.items() if v is None]
    for aid in unresolved:
        a = get_json(f"{ESPN_CORE}/seasons/2026/athletes/{aid}")
        cache[aid] = (a or {}).get("displayName") or str(aid)
        time.sleep(0.15)

    for r in rows:
        r["player"] = cache.get(r["espn_athlete_id"])
    return rows


def _num(v):
    if v is None:
        return None
    try:
        return float(str(v).replace("+", ""))
    except ValueError:
        return None


def snapshot_props(days=2, window_hours=PROP_WINDOW_HOURS):
    """Prop lines for games starting soon, appended with a timestamp."""
    abbrev = team_abbrev_map()
    now = pd.Timestamp.utcnow()
    cache = _load_athlete_cache()
    rows = []

    for offset in range(days):
        date_str = (dt.date.today() + dt.timedelta(days=offset)).isoformat()
        games = schedule(date_str, abbrev)
        if not games:
            continue
        events = espn_events(date_str)
        pairs = match_events(games, events)

        for g in games:
            hours = (g["start_utc"] - now).total_seconds() / 3600.0
            if hours < -1 or hours > window_hours:
                continue
            ev = pairs.get(g["game_pk"])
            if ev is None:
                continue
            got = fetch_props(ev["event_id"], cache)
            for r in got:
                r.update({k: g[k] for k in
                          ("game_pk", "game_date", "start_utc",
                           "home_team", "away_team")})
                r["snapshot_utc"] = now
            rows += got
            print(f"    {g['away_team']} @ {g['home_team']}: {len(got)} props")
            time.sleep(0.3)

    _save_athlete_cache(cache)
    if not rows:
        print("  no games inside the prop window")
        return None

    new = pd.DataFrame(rows)
    if os.path.exists(PROPS_LOG):
        old = pd.read_parquet(PROPS_LOG)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(
        subset=["game_pk", "espn_athlete_id", "prop_type", "snapshot_utc"],
        keep="last")
    new.to_parquet(PROPS_LOG, index=False)
    print(f"  props log now {len(new):,} rows across "
          f"{new['game_pk'].nunique():,} games -> {PROPS_LOG}")
    return new


def american_to_prob(o):
    o = pd.to_numeric(o, errors="coerce")
    return (-o / (-o + 100.0)).where(o < 0, 100.0 / (o + 100.0))


def build_closing():
    """Last snapshot strictly before first pitch, per game and provider."""
    if not os.path.exists(ODDS_LOG):
        raise SystemExit(f"no snapshot log at {ODDS_LOG}; run without --closing first")

    df = pd.read_parquet(ODDS_LOG)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)

    # Pre-game only, and never the live-odds feed. ESPN exposes a separate
    # "DraftKings - Live Odds" provider whose numbers move during the game --
    # a total of 4.5 in the sixth inning is not a closing total, and letting it
    # through would silently poison the benchmark with in-game information.
    pre = df[(df["snapshot_utc"] < df["start_utc"])
             & (~df["provider"].str.contains("Live", case=False, na=False))
             & (df["status"].isin(["Preview", "Scheduled", "Pre-Game"]))].copy()
    pre = pre.sort_values("snapshot_utc")
    close = pre.groupby(["game_pk", "provider"], as_index=False).last()

    p_home = american_to_prob(close["ml_home"])
    p_away = american_to_prob(close["ml_away"])
    total = p_home + p_away
    close["vig"] = total - 1.0
    close["p_home_close"] = p_home / total
    close["p_away_close"] = p_away / total
    close["lead_minutes"] = (
        (close["start_utc"] - close["snapshot_utc"]).dt.total_seconds() / 60)

    close.to_parquet(CLOSING, index=False)
    print(f"closing lines for {close['game_pk'].nunique():,} games -> {CLOSING}")
    if len(close):
        print(f"  median capture lead time: "
              f"{close['lead_minutes'].median():.0f} minutes before first pitch")
        print(f"  mean vig: {close['vig'].mean() * 100:.2f}%")
    return close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2,
                    help="how many days ahead to snapshot (today = 1)")
    ap.add_argument("--closing", action="store_true",
                    help="derive closing lines from the existing log")
    ap.add_argument("--no-props", action="store_true",
                    help="skip player props (game lines only)")
    args = ap.parse_args()

    if args.closing:
        build_closing()
        return

    print(f"snapshot at {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC")
    snapshot(args.days)
    if not args.no_props:
        print("props:")
        try:
            snapshot_props(args.days)
        except Exception as exc:      # noqa: BLE001 - never break game lines
            print(f"  prop capture failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
