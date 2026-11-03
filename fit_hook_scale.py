"""Find the hazard scale that makes starts the right length.

The 2025 prop backtest projects 15.12 outs from a starter against 15.66 actual
-- half an out short, uniformly across workload bands. That is worth about 0.17
strikeouts a start, and it was hiding the strikeout rate bias rather than being
independent of it: the model over-projected the rate and under-projected the
opportunity, and the two nearly cancelled in the average while both were wrong.

Correcting only the rate would have turned a -0.04 K bias into -0.19 and made
the unders lean worse. So both move together or neither does.

`hazard_scale` multiplies the fitted removal hazard; below 1 it lengthens
starts. This sweeps it against actual starter workload.

    python fit_hook_scale.py --season 2025 --games 220
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate_sim
import outcome_model as om
import props
import simulate as sim

PATH = os.path.join(config.PROCESSED_DIR, "hook_scale.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=220)
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()

    pa = om.load_pa()
    ps = props.PropSimulator(pa, args.season)
    games = evaluate_sim.extract_games(pa, args.season)
    np.random.default_rng(3).shuffle(games)
    games = games[:args.games]

    # Actual starter workload on the same games, so the target is not a
    # season-wide average being compared against a particular sample.
    keys = {(g["game_pk"]) for g in games}
    st = pa[pa["is_pa"] & pa["is_starter"] & pa["game_pk"].isin(keys)]
    actual = (st.groupby(["game_pk", "pitcher"])["outs_pa"].sum().mean())
    print(f"actual starter outs on these {len(games)} games: {actual:.3f}")

    rows = []
    for scale in (1.00, 0.90, 0.82, 0.75, 0.68):
        rng = np.random.default_rng(5)
        ps.hook.hazard_scale = scale
        tot = []
        for g in games:
            d = ps.run(g, args.sims, rng)
            tot.append(float(d["starter_outs"].mean())
                       if "starter_outs" in d else np.nan)
        m = float(np.nanmean(tot))
        rows.append((scale, m))
        print(f"  hazard_scale {scale:.2f}   projected starter outs {m:.3f}   "
              f"({m - actual:+.3f})")

    best = min(rows, key=lambda r: abs(r[1] - actual))
    print(f"\nclosest: hazard_scale {best[0]:.2f} -> {best[1]:.3f} "
          f"vs {actual:.3f} actual")
    with open(PATH, "w") as fh:
        json.dump({"season_fit": args.season, "hazard_scale": best[0],
                   "target_outs": actual, "achieved": best[1]}, fh, indent=2)
    print(f"wrote {PATH}")


if __name__ == "__main__":
    main()
