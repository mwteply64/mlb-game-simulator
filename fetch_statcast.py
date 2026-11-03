"""Resumable Statcast pitch-level download.

Pulls each season in half-month chunks and caches them as parquet. Re-running
skips any chunk already on disk, so the job survives interruption and can be
re-run each day during the season to pick up new games.

    python fetch_statcast.py                 # all seasons in config.SEASONS
    python fetch_statcast.py 2025 2026       # just these
    python fetch_statcast.py --refresh-last  # redownload the final chunk of
                                             # each season (in-progress months)
"""

import datetime as dt
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import pybaseball as pb

import config


def half_month_chunks(year):
    """Split a season into (start, end) date-string pairs of ~15 days."""
    start = dt.date.fromisoformat(f"{year}-{config.SEASON_START_MMDD}")
    end = dt.date.fromisoformat(f"{year}-{config.SEASON_END_MMDD}")
    today = dt.date.today()
    if end > today:
        end = today

    chunks = []
    cursor = start
    while cursor <= end:
        if cursor.day < 16:
            stop = cursor.replace(day=15)
        else:
            nxt = (cursor.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            stop = nxt - dt.timedelta(days=1)
        if stop > end:
            stop = end
        chunks.append((cursor.isoformat(), stop.isoformat()))
        cursor = stop + dt.timedelta(days=1)
    return chunks


def fetch_chunk(start, end):
    """One Statcast range pull, trimmed to regular-season rows and our columns."""
    df = pb.statcast(start_dt=start, end_dt=end, verbose=False)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=config.STATCAST_COLUMNS)

    df = df[df["game_type"] == "R"]
    keep = [c for c in config.STATCAST_COLUMNS if c in df.columns]
    missing = [c for c in config.STATCAST_COLUMNS if c not in df.columns]
    if missing:
        print(f"    warning: absent columns {missing}", flush=True)
    return df[keep].copy()


def fetch_season(year, refresh_last=False, force=False):
    chunks = half_month_chunks(year)
    for i, (start, end) in enumerate(chunks):
        path = config.raw_chunk_path(year, i)
        is_last = i == len(chunks) - 1
        # A chunk written on or before its window's last day was cut short by
        # `today` and is missing the rest of the window. Skipping it because the
        # file exists silently dropped 217 games from 2026 (Aug 15, Aug 29-31,
        # Sep 4-15) until the full-season market test came up short.
        written = (dt.date.fromtimestamp(os.path.getmtime(path))
                   if os.path.exists(path) else None)
        partial = written is not None and written <= dt.date.fromisoformat(end)
        if (written is not None and not force and not partial
                and not (refresh_last and is_last)):
            continue

        t0 = time.time()
        try:
            df = fetch_chunk(start, end)
        except Exception as exc:  # noqa: BLE001 - keep the long job alive
            print(f"  {year} chunk {i:02d} {start}..{end} FAILED: {exc}", flush=True)
            continue

        df.to_parquet(path, index=False, compression="zstd")
        print(
            f"  {year} chunk {i:02d} {start}..{end}: {len(df):>7,} pitches "
            f"in {time.time() - t0:5.1f}s",
            flush=True,
        )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh_last = "--refresh-last" in sys.argv
    # --force redownloads chunks that already exist. Needed when the retained
    # column set in config changes, since cached chunks are missing the new
    # columns entirely.
    force = "--force" in sys.argv
    seasons = [int(a) for a in args] if args else config.SEASONS

    os.makedirs(config.RAW_DIR, exist_ok=True)
    print(f"fetch_statcast start {dt.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"seasons: {seasons}  refresh_last={refresh_last}  force={force}",
          flush=True)

    overall = time.time()
    for year in seasons:
        print(f"[{year}]", flush=True)
        fetch_season(year, refresh_last=refresh_last, force=force)
    print(f"done in {(time.time() - overall) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
