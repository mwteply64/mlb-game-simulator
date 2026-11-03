"""Trim the pinch-hit hazard intercept until the simulated rate matches reality.

The fitted hazard is well calibrated per plate appearance, but the quantity that
matters is the share of plate appearances *taken* by a substitute, and the two
are not the same thing. Once a slot turns over it stays turned over, so the
observed share depends on when substitutions happen, not just how often the
hazard fires. Fitting gave 4.10% simulated against 3.74% real on 2025 -- right
shape, about ten percent too many.

Rather than solve that analytically, this measures and corrects, the same way
`fit_sim_calibration.py` does. The intercept moves by log(target / achieved)
each round, which is the right first-order step for a rare event, and the result
is re-measured rather than assumed.

Games and seed are held fixed across iterations so successive measurements
differ only by the intercept being tested, not by which games were drawn.

    python calibrate_pinch.py --season 2025
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import config
import evaluate_sim
import outcome_model as om
import pinch_hitting as ph
import props


def actual_rate(pa, season):
    """Share of plate appearances taken by a substitute, from the real data."""
    d = pa[pa["is_pa"] & (pa["season"] == season)]
    key = ["game_pk", "batting_team", "lineup_slot"]
    first = d.sort_values("at_bat_number").drop_duplicates(subset=key,
                                                           keep="first")
    starters = set(zip(first["game_pk"], first["batting_team"],
                       first["lineup_slot"], first["batter"]))
    is_sub = [(g, t, s, b) not in starters for g, t, s, b in
              zip(d["game_pk"], d["batting_team"], d["lineup_slot"],
                  d["batter"])]
    return float(np.mean(is_sub))


def measure(ps, games, n_sims, weights, li, seed=11):
    ps.pinch = (weights, li)
    rng = np.random.default_rng(seed)
    sub = tot = 0
    for g in games:
        d = ps.run(g, n_sims, rng)
        sub += int(d["sub_pa"][0])
        tot += int(d["sub_pa"][1])
    return sub / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    import leverage as lev
    pa = om.load_pa()
    target = actual_rate(pa, args.season)
    print(f"target substitute share on {args.season}: {target * 100:.3f}%")

    blob = json.load(open(ph.PATH))
    w = np.array(blob["weights"], dtype=float)
    base_intercept = float(blob.get("intercept_fitted", w[0]))
    li = lev.load_leverage_table()

    ps = props.PropSimulator(pa, args.season)
    games = evaluate_sim.extract_games(pa, args.season)
    np.random.default_rng(3).shuffle(games)
    games = games[:args.games]

    for it in range(args.iters):
        got = measure(ps, games, args.sims, w, li)
        print(f"  iter {it + 1}: intercept {w[0]:+.4f} -> "
              f"{got * 100:.3f}%  ({(got - target) * 100:+.3f} pp)")
        if abs(got - target) < 0.0005:
            print("  within half a tenth of a point; stopping")
            break
        w[0] += float(np.log(target / max(got, 1e-9)))

    blob["weights"] = w.tolist()
    blob["intercept_fitted"] = base_intercept
    blob["intercept_calibrated_on"] = args.season
    blob["note"] = ("intercept shifted so the simulated share of substitute "
                    "plate appearances matches the season's actual share; the "
                    "per-PA fit is unchanged in every other coefficient")
    with open(ph.PATH, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwrote {ph.PATH}  (intercept {base_intercept:+.4f} -> {w[0]:+.4f})")


if __name__ == "__main__":
    main()
