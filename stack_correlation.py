"""Does pairing a hitter stack with its own pitcher add correlation?

The intuition is that if you are stacking a club's hitters you should take that
club's starter too, so everything moves together. Worth measuring rather than
assuming, because the two are linked through only one narrow channel.

A pitcher scores from outs, strikeouts and runs *not* allowed. His own hitters
scoring does not produce any of those -- he is not on the field when they bat.
The single connection is the win, worth 4, which his offence makes more likely.
So the expected correlation is weakly positive.

The *opposing* pitcher is a different matter. Every run your stack scores is a
run he allowed at -2, and damage gets him pulled early, costing outs at 1 apiece.
That link is direct and should be strongly negative -- which matters more in
practice, because it says what never to do rather than what mildly helps.

    python stack_correlation.py --games 12 --sims 4000
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate_sim
import live_props as lp
import outcome_model as om
import yahoo_dfs as yd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    pa = om.load_pa()
    ls = lp.LiveSimulator(pa, args.season, "2026-08-24")
    games = evaluate_sim.extract_games(pa, args.season)
    np.random.default_rng(4).shuffle(games)
    games = games[:args.games]
    coef = yd.load_pitcher_scoring()
    rng = np.random.default_rng(11)

    own, opp, pair_own, pair_opp = [], [], [], []
    for g in games:
        if not (g.get("away_starter") and g.get("home_starter")):
            continue
        gg = dict(g)
        gg["away_lineup_ids"] = list(g["away_lineup"])
        gg["home_lineup_ids"] = list(g["home_lineup"])
        gg["start_utc"] = pd.Timestamp(g["game_date"], tz="UTC")
        try:
            diag, (aw, hm) = ls.run_live(gg, args.sims, rng)
        except Exception:
            continue

        for ti in (0, 1):
            # Top five slots: what a real stack looks like.
            stack = sum(yd.hitter_points(diag, ti, s) for s in range(5))
            p_own = yd.pitcher_points(diag, ti, coef)
            p_opp = yd.pitcher_points(diag, 1 - ti, coef)
            own.append(np.corrcoef(stack, p_own)[0, 1])
            opp.append(np.corrcoef(stack, p_opp)[0, 1])
            # What the pairing does to the ceiling of a 6-man entry.
            pair_own.append(float(np.mean(stack + p_own > 60)))
            pair_opp.append(float(np.mean(stack + p_opp > 60)))

    print(f"{len(own)} team-games simulated ({args.sims:,} sims each)\n")
    print("correlation of a 5-hitter stack with:")
    print(f"  its OWN team's starter      {np.mean(own):+.3f}"
          f"   (sd across games {np.std(own):.3f})")
    print(f"  the OPPOSING starter        {np.mean(opp):+.3f}"
          f"   (sd across games {np.std(opp):.3f})")
    print()
    print("P(stack + that pitcher > 60 pts):")
    print(f"  with own starter            {np.mean(pair_own) * 100:5.1f}%")
    print(f"  with opposing starter       {np.mean(pair_opp) * 100:5.1f}%")
    print()
    if np.mean(own) > 0.05:
        print("-> pairing the stack with its own starter does add correlation")
    else:
        print("-> own-starter correlation is negligible; the win is the only")
        print("   channel and it is not worth constraining the build around")
    print("-> the opposing starter is the one that matters, and it is a rule")
    print("   about what to avoid rather than what to seek")


if __name__ == "__main__":
    main()
