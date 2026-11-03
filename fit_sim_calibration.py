"""End-to-end outcome-rate correction for the simulator.

The simulator over-produces extra-base hits: on 2024, doubles +4.4%, triples
+11.7%, home runs +3.8%, while singles, strikeouts and outs land within one
percent. That inflates runs by about 0.3 per game -- invisible in win
probability, which cancels across both teams, but a direct bias on totals.

Four candidate mechanisms were tested and none accounted for it: the league
nowcast (already optimal), the transition window (12% of the gap), times
through the order (already inside the existing correction), and park factors
(no effect). So this measures the error where it is actually observable -- in
the rates the simulator produces -- and corrects it there.

**This is a patch, not a fix.** The mechanism remains unidentified, so the
correction is fitted to a bias whose cause could itself vary. It must be
re-fitted whenever the model changes, and it should be checked on a season it
was not fitted to, which is what the verify step is for.

    python fit_sim_calibration.py --fit 2024
    python fit_sim_calibration.py --verify 2025
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

OUT = config.OUTCOMES
PATH = os.path.join(config.PROCESSED_DIR, "sim_calibration.json")


def measure(season, n_games, n_sims, seed=11, sample_seed=11):
    """Outcome rates the simulator produces, against what the season did.

    `sample_seed` picks the games and is held fixed across iterations, so
    successive measurements differ only by the correction being tested rather
    than by which games were drawn.
    """
    import simulate as sim
    # The correction is read at import, so a freshly written one is invisible
    # to an already-loaded module. Re-read it before every measurement.
    sim.SIM_CALIBRATION = sim._load_sim_calibration()

    pa = om.load_pa()
    ps = props.PropSimulator(pa, season)
    games = evaluate_sim.extract_games(pa, season)
    np.random.default_rng(sample_seed).shuffle(games)
    games = games[:n_games]
    rng = np.random.default_rng(seed)

    totals = np.zeros(len(OUT), dtype=np.int64)
    sim_runs, act_runs = [], []
    for g in games:
        diag = ps.run(g, n_sims, rng)
        totals += diag["outcome_counts"]
        sim_runs.append(float(diag["bat_rbi"].sum(axis=(1, 2)).mean()))
        act_runs.append(g["home_runs"] + g["away_runs"])

    sim_rates = totals / totals.sum()
    d = pa[pa["is_pa"] & (pa["season"] == season) & pa["outcome"].isin(OUT)]
    act_rates = (d["outcome"].value_counts(normalize=True)
                 .reindex(OUT).fillna(0).to_numpy())
    return sim_rates, act_rates, float(np.mean(sim_runs)), float(np.mean(act_runs))


def report(sim_rates, act_rates, sim_r, act_r, label):
    print(f"\n{label}")
    print(f"  {'outcome':<9}{'simulated':>11}{'actual':>10}{'rel':>9}")
    for i, o in enumerate(OUT):
        rel = (sim_rates[i] - act_rates[i]) / act_rates[i] * 100 \
            if act_rates[i] else float("nan")
        print(f"  {o:<9}{sim_rates[i] * 100:>10.3f}%{act_rates[i] * 100:>9.3f}%"
              f"{rel:>+8.1f}%")
    print(f"  runs per game: simulated {sim_r:.3f}  actual {act_r:.3f}  "
          f"({sim_r - act_r:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int)
    ap.add_argument("--verify", type=int)
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    if args.fit:
        calib = np.ones(len(OUT))
        if os.path.exists(PATH):
            os.remove(PATH)          # fit from scratch, never on top of itself

        for it in range(args.iters):
            sim_rates, act_rates, sr, ar = measure(
                args.fit, args.games, args.sims, seed=11 + it)
            report(sim_rates, act_rates, sr, ar,
                   f"=== {args.fit}, iteration {it + 1} ===")

            # The simulator renormalises every vector, so only the relative
            # values matter -- no rescaling needed, and rescaling only made the
            # printed numbers harder to read against the raw ratios.
            calib = calib * (act_rates / np.maximum(sim_rates, 1e-9))
            with open(PATH, "w") as fh:
                json.dump({"season_fit": args.fit,
                           "calibration": calib.tolist(),
                           "outcomes": OUT}, fh, indent=2)
            print("  correction now: " +
                  "  ".join(f"{o} {c:.4f}" for o, c in zip(OUT, calib)))

        print(f"\nwrote {PATH}")
        print("Verify on a season this was not fitted to before trusting it.")

    if args.verify:
        if not os.path.exists(PATH):
            raise SystemExit("no calibration fitted yet")
        blob = json.load(open(PATH))
        print(f"applying correction fitted on {blob['season_fit']}")
        sim_rates, act_rates, sr, ar = measure(
            args.verify, args.games, args.sims, seed=99)
        report(sim_rates, act_rates, sr, ar,
               f"=== {args.verify} (held out) ===")


if __name__ == "__main__":
    main()
