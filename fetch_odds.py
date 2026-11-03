"""Download and normalize historical MLB betting odds.

Source: sportsbookreviewsonline.com season archives. Note that this archive
stops at 2021 and the site states it will not be updated, so recent seasons
must come from elsewhere -- see README. The parsing, de-vigging and join logic
here is source-agnostic: point `load_raw` at any file with the same two-rows-
per-game layout and the rest works.

Layout of the source files: two rows per game, visitor first then home, with
open/close moneylines and the run total split across the pair.

    python fetch_odds.py 2021
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config

ODDS_DIR = os.path.join(config.RAW_DIR, "odds")
BASE_URL = ("https://www.sportsbookreviewsonline.com/wp-content/uploads/"
            "sportsbookreviewsonline_com_737/mlb-odds-{year}.xlsx")

# Archive abbreviations differ from Statcast's in a handful of cases.
TEAM_FIXES = {
    "ANA": "LAA", "ARI": "AZ", "CHW": "CWS", "CUB": "CHC", "CUP": "CHC",
    "FLA": "MIA", "KAN": "KC", "LOS": "LAD", "NYM": "NYM", "NYY": "NYY",
    "SDG": "SD", "SFO": "SF", "TAM": "TB", "TAB": "TB", "WAS": "WSH",
    "OAK": "ATH",
}


def download(year):
    os.makedirs(ODDS_DIR, exist_ok=True)
    path = os.path.join(ODDS_DIR, f"mlb-odds-{year}.xlsx")
    if os.path.exists(path):
        print(f"  {year}: cached")
        return path

    url = BASE_URL.format(year=year)
    r = requests.get(url, timeout=60,
                     headers=None)   # see record_odds.py: no spoofed UA
    r.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(r.content)
    print(f"  {year}: {len(r.content) / 1024:.0f} KB -> {path}")
    return path


def american_to_prob(odds):
    """American moneyline to implied probability, vig included."""
    o = pd.to_numeric(odds, errors="coerce")
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def devig(p_away, p_home):
    """Proportional de-vig: scale both sides so they sum to one.

    The two implied probabilities sum to more than 1 -- that excess is the
    book's margin. Normalizing is the standard treatment and is accurate
    enough here; Shin and the power method are refinements that would not
    change any conclusion at this resolution.
    """
    total = p_away + p_home
    ok = np.isfinite(total) & (total > 0)
    return (np.where(ok, p_away / total, np.nan),
            np.where(ok, p_home / total, np.nan))


def load_raw(path, year):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize(df, year):
    """Collapse the two-rows-per-game layout into one row per game."""
    need = {"Date", "VH", "Team", "Final", "Open", "Close"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"unexpected columns; missing {missing}\n"
                         f"have: {list(df.columns)}")

    df = df.copy()
    df["Team"] = df["Team"].astype(str).str.strip().str.upper()
    df["Team"] = df["Team"].replace(TEAM_FIXES)

    # Rows alternate visitor/home. Pair them positionally rather than by team,
    # which keeps doubleheaders intact.
    df["pair"] = np.arange(len(df)) // 2
    away = df[df["VH"].astype(str).str.upper().str.startswith("V")]
    home = df[df["VH"].astype(str).str.upper().str.startswith("H")]

    m = away.merge(home, on="pair", suffixes=("_a", "_h"))

    out = pd.DataFrame({
        "season": year,
        "date_raw": m["Date_a"],
        "away_team": m["Team_a"],
        "home_team": m["Team_h"],
        "away_runs": pd.to_numeric(m["Final_a"], errors="coerce"),
        "home_runs": pd.to_numeric(m["Final_h"], errors="coerce"),
        "ml_open_away": pd.to_numeric(m["Open_a"], errors="coerce"),
        "ml_open_home": pd.to_numeric(m["Open_h"], errors="coerce"),
        "ml_close_away": pd.to_numeric(m["Close_a"], errors="coerce"),
        "ml_close_home": pd.to_numeric(m["Close_h"], errors="coerce"),
    })

    pa_, ph_ = american_to_prob(out["ml_close_away"]), american_to_prob(out["ml_close_home"])
    out["vig"] = pa_ + ph_ - 1.0
    out["p_away_close"], out["p_home_close"] = devig(pa_, ph_)

    pa_o, ph_o = american_to_prob(out["ml_open_away"]), american_to_prob(out["ml_open_home"])
    out["p_home_open"] = devig(pa_o, ph_o)[1]

    out["home_won"] = (out["home_runs"] > out["away_runs"]).astype(float)
    return out


def parse_dates(out, year):
    """The archive stores dates as MMDD integers that roll over the year."""
    raw = pd.to_numeric(out["date_raw"], errors="coerce")
    mm = (raw // 100).astype("Int64")
    dd = (raw % 100).astype("Int64")
    out["game_date"] = pd.to_datetime(
        {"year": year, "month": mm, "day": dd}, errors="coerce")
    return out


def join_to_games(odds, games):
    """Attach closing probabilities to games by date, teams, and final score.

    Date and teams alone are ambiguous on doubleheader dates, and the two
    sources do not order the pair the same way -- matching positionally
    mismatched roughly half of them, which showed up as 72 games where the
    joined final scores disagreed. The final score resolves the pair
    unambiguously, and both sources record it independently, so it is a safe
    join key rather than a circular one.

    `games` needs game_pk, game_date, home_team, away_team, home_runs,
    away_runs. Returns `games` with the odds columns attached, plus a
    `join_kind` column recording how each row matched.
    """
    o = odds.dropna(subset=["game_date"]).copy()
    o["game_date"] = pd.to_datetime(o["game_date"])
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])

    odds_cols = ["p_home_close", "p_away_close", "p_home_open", "vig",
                 "ml_close_home", "ml_close_away"]

    strict_key = ["game_date", "home_team", "away_team",
                  "home_runs", "away_runs"]
    strict = g.merge(o[strict_key + odds_cols].drop_duplicates(strict_key),
                     on=strict_key, how="left")
    strict["join_kind"] = np.where(strict["p_home_close"].notna(),
                                   "date+teams+score", None)

    # Anything left over: fall back to date and teams where that pair is
    # unique, which covers rescheduled games whose scores were recorded
    # differently.
    need = strict["p_home_close"].isna()
    if need.any():
        loose_key = ["game_date", "home_team", "away_team"]
        counts = o.groupby(loose_key).size().rename("n").reset_index()
        unique = counts[counts["n"] == 1][loose_key]
        loose = (o.merge(unique, on=loose_key)[loose_key + odds_cols]
                 .drop_duplicates(loose_key))
        filled = strict.loc[need, loose_key].merge(loose, on=loose_key,
                                                   how="left")
        for c in odds_cols:
            strict.loc[need, c] = filled[c].to_numpy()
        strict.loc[need & strict["p_home_close"].notna(),
                   "join_kind"] = "date+teams"

    return strict


def main():
    years = [int(a) for a in sys.argv[1:]] or [2021]
    frames = []
    print("downloading:")
    for y in years:
        path = download(y)
        df = normalize(load_raw(path, y), y)
        df = parse_dates(df, y)
        frames.append(df)

    odds = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(config.PROCESSED_DIR, "odds.parquet")
    odds.to_parquet(out_path, index=False)
    print(f"\nwrote {len(odds):,} games -> {out_path}")

    good = odds.dropna(subset=["p_home_close", "home_won"])
    print(f"  usable closing lines: {len(good):,}")
    print(f"  mean vig: {good['vig'].mean() * 100:.2f}%")
    print(f"  home win rate: {good['home_won'].mean() * 100:.2f}%")
    print(f"  mean de-vigged home probability: "
          f"{good['p_home_close'].mean() * 100:.2f}%")

    eps = 1e-9
    p = np.clip(good["p_home_close"].to_numpy(), eps, 1 - eps)
    y = good["home_won"].to_numpy()
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = np.mean((p - y) ** 2)
    acc = ((p > 0.5) == (y > 0.5)).mean()
    base = y.mean()
    ll_base = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))

    print(f"\n  MARKET (closing, de-vigged):")
    print(f"    log loss {ll:.5f}   Brier {br:.5f}   accuracy {acc * 100:.2f}%")
    print(f"    always-home baseline log loss {ll_base:.5f}, "
          f"accuracy {max(base, 1 - base) * 100:.2f}%")

    po = np.clip(good["p_home_open"].to_numpy(), eps, 1 - eps)
    llo = -np.mean(y * np.log(po) + (1 - y) * np.log(1 - po))
    print(f"    opening line log loss {llo:.5f} "
          f"(closing beats opening by {llo - ll:+.5f})")


if __name__ == "__main__":
    main()
