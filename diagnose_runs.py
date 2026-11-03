"""Split a run-level bias into its two possible causes.

If the simulated outcome RATES match the season's real ones but the runs do
not, the state machine converts outcomes to runs too generously. If the rates
themselves are off, the projection or context layer is at fault, and whichever
outcome is furthest out says which.

Also reports the bias by month, because the projections use month-boundary
as-of models: an April game has no data from its own season at all and must
project from prior years. A season that is genuinely lower-scoring than its
predecessors -- new ball, rule change, weather -- cannot be anticipated in
April, only learned. If that is what is happening, the bias should be large
early and shrink as the nowcast fills in.

    python diagnose_runs.py --season 2024 --games 400
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate_sim
import outcome_model as om
import props

OUT = config.OUTCOMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pa = om.load_pa()
    ps = props.PropSimulator(pa, args.season)
    games = evaluate_sim.extract_games(pa, args.season)
    rng.shuffle(games)
    games = games[:args.games]
    print(f"{args.season}: {len(games)} games x {args.sims} simulations\n")

    totals = np.zeros(len(OUT), dtype=np.int64)
    rows = []
    for i, g in enumerate(games):
        diag = ps.run(g, args.sims, rng)
        totals += diag["outcome_counts"]
        sim_runs = float((diag["bat_rbi"].sum(axis=(1, 2))).mean())
        rows.append({"game_pk": g["game_pk"],
                     "month": pd.Timestamp(g["game_date"]).month,
                     "sim_runs": sim_runs,
                     "act_runs": g["home_runs"] + g["away_runs"]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(games)}")

    sim_rates = totals / totals.sum()

    d = pa[pa["is_pa"] & (pa["season"] == args.season)
           & pa["outcome"].isin(OUT)]
    act_rates = (d["outcome"].value_counts(normalize=True)
                 .reindex(OUT).fillna(0).to_numpy())

    print("\n=== outcome rates: simulated vs actual ===")
    print(f"  {'outcome':<9}{'simulated':>11}{'actual':>10}{'diff':>9}{'rel':>9}")
    for i, o in enumerate(OUT):
        diff = sim_rates[i] - act_rates[i]
        rel = diff / act_rates[i] * 100 if act_rates[i] else float("nan")
        print(f"  {o:<9}{sim_rates[i] * 100:>10.3f}%{act_rates[i] * 100:>9.3f}%"
              f"{diff * 100:>+9.3f}{rel:>+8.1f}%")

    on_base = [OUT.index(o) for o in ("BB", "HBP", "1B", "2B", "3B", "HR")]
    print(f"\n  reaching base: simulated {sim_rates[on_base].sum() * 100:.3f}%  "
          f"actual {act_rates[on_base].sum() * 100:.3f}%  "
          f"({(sim_rates[on_base].sum() - act_rates[on_base].sum()) * 100:+.3f})")

    r = pd.DataFrame(rows)
    print("\n=== run bias by month ===")
    print(f"  {'month':<7}{'n':>5}{'simulated':>11}{'actual':>9}{'diff':>9}")
    for m, grp in r.groupby("month"):
        print(f"  {m:<7}{len(grp):>5}{grp['sim_runs'].mean():>11.3f}"
              f"{grp['act_runs'].mean():>9.3f}"
              f"{grp['sim_runs'].mean() - grp['act_runs'].mean():>+9.3f}")
    print(f"\n  overall: simulated {r['sim_runs'].mean():.3f}   "
          f"actual {r['act_runs'].mean():.3f}   "
          f"({r['sim_runs'].mean() - r['act_runs'].mean():+.3f} per game, "
          f"{(r['sim_runs'].mean() - r['act_runs'].mean()) / 2:+.3f} per team)")
    print("\n  A bias that shrinks through the season is the nowcast learning")
    print("  a run environment it could not have known in April. A flat one is")
    print("  a model defect.")


if __name__ == "__main__":
    main()
