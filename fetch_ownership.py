"""Actual ownership, from Yahoo's own contest entries.

The field-simulation work assumed ownership had to be proxied from salary and
Yahoo's published projections, because ownership feeds are normally sold. They
do not have to be: Yahoo exposes every entrant's roster.

    /v2/contestEntries?contestId=   -> entry ids, paginated
    /v2/contestEntry/{entry_id}     -> lineupSlotList, the actual roster

So real ownership is measurable for any contest, which matters three ways:

* the field simulation can be checked against a real field rather than a model,
* the ownership proxy can be calibrated against truth instead of assumed,
* and leverage becomes computable -- a player we like whom the field does not
  is worth more than his projection alone says, which is the whole basis of
  large-field play.

One request per entry, so a 360-entry contest costs 360 calls. Fine for a
handful of contests, and the results are cached so a contest is only ever
fetched once.

    python fetch_ownership.py --contest 15744639
    python fetch_ownership.py --recent 5
"""

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config

BASE = "https://dfyql-ro.sports.yahoo.com/v2"
PATH = os.path.join(config.RAW_DIR, "ownership.parquet")


def entry_ids(contest_id, page_size=100, pause=0.15):
    """Every entry id in a contest, following pagination."""
    out, start = [], 0
    while True:
        r = requests.get(f"{BASE}/contestEntries",
                         params={"contestId": contest_id, "start": start,
                                 "limit": page_size}, timeout=30)
        if r.status_code != 200:
            break
        j = r.json()
        e = j.get("entries")
        e = e.get("result") if isinstance(e, dict) else e
        if not e:
            break
        out += [int(x["id"]) for x in e]
        pg = (j.get("pagination") or {})
        pg = pg.get("result") if isinstance(pg, dict) else pg
        total = int((pg or {}).get("totalCount", len(out)))
        start += len(e)
        time.sleep(pause)
        if start >= total or not len(e):
            break
    return out


def entry_roster(entry_id, pause=0.12):
    """The players in one entry."""
    r = requests.get(f"{BASE}/contestEntry/{entry_id}", timeout=30)
    time.sleep(pause)
    if r.status_code != 200:
        return None
    j = r.json()
    e = j.get("entries")
    e = e.get("result") if isinstance(e, dict) else e
    if not e:
        return None
    x = e[0]
    rows = []
    for s in x.get("lineupSlotList", []):
        # `player` is a SIBLING of `lineupSlot`, not nested inside it, and is
        # absent for an unfilled slot. Team sits one level down again, at
        # player.team.abbr.
        slot = (s.get("lineupSlot") or {}).get("abbr")
        pl = s.get("player")
        if not pl:
            continue
        nm = f"{pl.get('firstName', '')} {pl.get('lastName', '')}".strip()
        if not nm:
            continue
        rows.append({"entry_id": int(entry_id), "slot": slot, "name": nm,
                     "code": pl.get("code"),
                     "salary": pl.get("salary"),
                     "points": pl.get("points"),
                     "team": (pl.get("team") or {}).get("abbr")})
    return rows


def fetch_contest(contest_id, limit=None, verbose=True):
    ids = entry_ids(contest_id)
    if limit:
        ids = ids[:limit]
    if verbose:
        print(f"contest {contest_id}: {len(ids)} entries")
    rows = []
    for k, eid in enumerate(ids, 1):
        r = entry_roster(eid)
        if r:
            for x in r:
                x["contest_id"] = int(contest_id)
            rows += r
        if verbose and k % 25 == 0:
            print(f"  {k}/{len(ids)}")
    return pd.DataFrame(rows)


def ownership(df):
    """Share of entries rostering each player."""
    n = df["entry_id"].nunique()
    o = (df.groupby("name")["entry_id"].nunique() / max(n, 1) * 100)
    return o.sort_values(ascending=False), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contest", type=int, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    df = fetch_contest(args.contest, args.limit)
    if not len(df):
        raise SystemExit("no rosters returned")

    if os.path.exists(PATH):
        df = pd.concat([pd.read_parquet(PATH), df], ignore_index=True)
        df = df.drop_duplicates(subset=["contest_id", "entry_id", "name"])
    df.to_parquet(PATH, index=False)

    o, n = ownership(df[df["contest_id"] == args.contest])
    print(f"\nactual ownership across {n} entries:")
    for nm, pct in o.head(15).items():
        print(f"  {nm:<24} {pct:5.1f}%")
    print(f"\nwrote {len(df):,} roster spots -> {PATH}")


if __name__ == "__main__":
    main()
