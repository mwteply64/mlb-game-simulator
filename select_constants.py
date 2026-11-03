"""Select the dispersion and expansion constants on 2024, honestly.

Both were originally chosen by evaluating candidates on 2025 and keeping the
winner -- which is selection on the evaluation set, and inflates the reported
2025 numbers by an unknown amount. This re-runs the choice using 2024 only, so
2025 becomes a clean holdout that no shipped constant has seen.

The two parameters get different objectives, deliberately:

* **Dispersion sigma** is matched to the run-distribution spread. It is one
  parameter fitted to one statistic (the team-runs standard deviation ratio),
  which barely overfits and is not a predictive target at all.
* **Expansion** is chosen on held-out log loss, which is a predictive target and
  is exactly where selection bias lives. Doing it on 2024 is the whole point.

    python select_constants.py --stage baseline
    python select_constants.py --stage dispersion
    python select_constants.py --stage expansion
"""

import argparse
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

SELECT_SEASON = 2024
PY = sys.executable


def run_sim(season, sims, dispersion, expand, suffix):
    cmd = [PY, "evaluate_sim.py", "--season", str(season), "--sims", str(sims),
           "--dispersion", str(dispersion), "--expand", str(expand),
           "--suffix", suffix]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=config.PROJECT_DIR)
    return out.stdout


def parse(stdout):
    """Pull the numbers that matter out of evaluate_sim's report."""
    res = {}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("std dev   simulated"):
            parts = s.split()
            res["sim_sd"] = float(parts[3])
            res["act_sd"] = float(parts[5])
            res["sd_ratio"] = float(parts[7])
        elif s.startswith("mean      simulated"):
            parts = s.split()
            res["sim_mean"] = float(parts[2])
            res["act_mean"] = float(parts[4])
        elif s.startswith("model     log loss"):
            parts = s.split()
            res["log_loss"] = float(parts[3])
            res["brier"] = float(parts[5])
        elif s.startswith("accuracy"):
            res["accuracy"] = float(s.split()[1].rstrip("%"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["baseline", "dispersion", "expansion", "all"])
    ap.add_argument("--sims", type=int, default=600)
    args = ap.parse_args()

    if args.stage in ("baseline", "all"):
        print(f"=== baseline run for {SELECT_SEASON} "
              f"(expansion needs a prior run to read team strength from) ===")
        out = run_sim(SELECT_SEASON, args.sims, 0.0, 1.0, "")
        r = parse(out)
        print(f"  mean {r.get('sim_mean')} vs {r.get('act_mean')}   "
              f"sd ratio {r.get('sd_ratio')}   log loss {r.get('log_loss')}")

    if args.stage in ("dispersion", "all"):
        print(f"\n=== dispersion sigma, selected on {SELECT_SEASON} ===")
        print(f"  {'sigma':>6} {'sd ratio':>10} {'mean':>8} {'log loss':>10}")
        best, best_gap = None, 9e9
        for sigma in (0.0, 0.10, 0.15, 0.19, 0.24, 0.30):
            r = parse(run_sim(SELECT_SEASON, args.sims, sigma, 1.0, "_sel"))
            gap = abs(r.get("sd_ratio", 0) - 1.0)
            flag = ""
            if gap < best_gap:
                best, best_gap, flag = sigma, gap, "  <-- closest to 1.0"
            print(f"  {sigma:>6.2f} {r.get('sd_ratio', float('nan')):>10.4f} "
                  f"{r.get('sim_mean', float('nan')):>8.3f} "
                  f"{r.get('log_loss', float('nan')):>10.5f}{flag}")
        print(f"\n  selected sigma = {best}")

    if args.stage in ("expansion", "all"):
        print(f"\n=== expansion factor, selected on {SELECT_SEASON} ===")
        print(f"  {'expand':>7} {'log loss':>10} {'Brier':>9} {'accuracy':>10} "
              f"{'sd ratio':>10}")
        best, best_ll = None, 9e9
        for ex in (1.0, 1.1, 1.2, 1.35, 1.5):
            r = parse(run_sim(SELECT_SEASON, args.sims, 0.19, ex, "_sel"))
            ll = r.get("log_loss", 9e9)
            flag = ""
            if ll < best_ll:
                best, best_ll, flag = ex, ll, "  <-- best"
            print(f"  {ex:>7.2f} {ll:>10.5f} {r.get('brier', float('nan')):>9.5f} "
                  f"{r.get('accuracy', float('nan')):>9.2f}% "
                  f"{r.get('sd_ratio', float('nan')):>10.4f}{flag}")
        print(f"\n  selected expansion = {best}")

    print("\nNow verify on 2025 with the selected values. That season has not")
    print("been used to choose anything, so the numbers it gives are clean.")


if __name__ == "__main__":
    main()
