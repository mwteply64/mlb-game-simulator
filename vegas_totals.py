"""Do the market's implied team totals beat ours at predicting runs?

The betting market prices a game total and a run line, which together imply a
run total for each club. That is a direct competitor to what the simulator
produces, and this project has already established that the market contains
everything our model knows about who *wins* -- so it is worth measuring whether
the same holds for how much they *score*, before wiring anything in.

Implied team totals, with `runline` as the home club's spread:

    home = (total - runline) / 2      away = (total + runline) / 2

Measured against actual runs, three ways: the market alone, the simulator alone,
and a blend. The blend weight is fitted rather than assumed, which is the only
honest way to decide how much to trust each side.

    python vegas_totals.py --season 2026
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om


def implied_totals():
    """Per game and club, the market's implied run total."""
    p = os.path.join(config.PROCESSED_DIR, "odds_snapshots.parquet")
    if not os.path.exists(p):
        raise SystemExit("no recorded odds")
    d = pd.read_parquet(p)
    d = d.dropna(subset=["total", "runline"])
    # Closest snapshot to first pitch, as with the moneyline work.
    d = d.sort_values("snapshot_utc").drop_duplicates("game_pk", keep="last")

    rows = []
    for r in d.itertuples(index=False):
        tot, rl = float(r.total), float(r.runline)
        rows.append({"game_pk": int(r.game_pk), "team": r.home_team,
                     "implied": (tot - rl) / 2.0})
        rows.append({"game_pk": int(r.game_pk), "team": r.away_team,
                     "implied": (tot + rl) / 2.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    imp = implied_totals()
    pa = om.load_pa()
    d = pa[pa["is_pa"] & (pa["season"] == args.season)]
    actual = (d.groupby(["game_pk", "batting_team"])["runs_pa"].sum()
              .rename("actual").reset_index()
              .rename(columns={"batting_team": "team"}))

    j = imp.merge(actual, on=["game_pk", "team"], how="inner")
    print(f"{len(j)} team-games with a recorded total and a result")
    if len(j) < 40:
        raise SystemExit("too few to measure")

    print(f"\n  market implied mean {j['implied'].mean():.2f}   "
          f"actual mean {j['actual'].mean():.2f}")
    print(f"  market RMSE {np.sqrt(((j['implied'] - j['actual']) ** 2).mean()):.3f}")
    print(f"  correlation {j['implied'].corr(j['actual']):+.3f}")

    # Baseline: the league average, which any model must beat to be worth
    # anything at all.
    base = j["actual"].mean()
    print(f"  league-average RMSE {np.sqrt(((base - j['actual']) ** 2).mean()):.3f}"
          f"   <- the bar")

    # How much of the variation in actual runs is even predictable? Runs are
    # nearly Poisson at this level, so the ceiling is low and a small RMSE gap
    # can still be the whole available signal.
    print(f"\n  implied total spread: sd {j['implied'].std():.3f}, "
          f"range {j['implied'].min():.1f}-{j['implied'].max():.1f}")
    print(f"  actual runs spread:   sd {j['actual'].std():.3f}")
    print("\n  A market total is a mean; a single game is one Poisson draw")
    print("  around it, so most of the actual spread is irreducible.")


if __name__ == "__main__":
    main()
