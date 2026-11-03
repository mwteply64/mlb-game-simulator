"""Platt-scale the simulator's win probabilities and decompose the loss.

Platt scaling fits two parameters on a *previous* season and applies them to the
holdout:

    p' = sigmoid(a * logit(p) + b)

It is a monotone transform of the output, which is exactly what makes it a
clean diagnostic. Because it cannot reorder games, it cannot change the
simulator's ability to tell a good matchup from a bad one -- AUC is invariant
under it. All it can fix is *how confident* the model is.

So the split is informative:

* If calibration recovers most of the gap to the other models, the simulator's
  ordering of games is sound and only its confidence was wrong.
* If it recovers little, the ordering itself is weak, and no transformation of
  the output will help -- that requires better inputs or better structure.

Murphy's decomposition of the Brier score into reliability (miscalibration) and
resolution (discrimination) is reported alongside, since it says the same thing
in a form that separates the two directly.

    python calibrate_sim.py --fit 2024 --apply 2025
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(p, y, iters=2000, lr=0.05):
    """Two-parameter logistic fit on the log-odds of the model's output."""
    z = logit(p)
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        ga = np.mean((q - y) * z)
        gb = np.mean(q - y)
        a -= lr * ga
        b -= lr * gb
    return a, b


def apply_platt(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def metrics(p, y, eps=1e-9):
    pc = np.clip(p, eps, 1 - eps)
    ll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    brier = np.mean((pc - y) ** 2)
    acc = ((pc > 0.5) == (y > 0.5)).mean()
    return ll, brier, acc


def auc(p, y):
    """Rank-based AUC. Invariant under any monotone transform of p, which is
    the point -- it shows Platt scaling changes confidence, not ordering."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks within ties so the invariance is exact.
    df = pd.DataFrame({"p": p, "r": ranks})
    ranks = df.groupby("p")["r"].transform("mean").to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def murphy(p, y, n_bins=10):
    """Brier = reliability - resolution + uncertainty.

    Reliability is miscalibration (lower is better, zero is perfect).
    Resolution is how far the model's bins depart from the base rate (higher is
    better); it is the part Platt scaling cannot create.
    """
    base = y.mean()
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        w = m.mean()
        pbar, ybar = p[m].mean(), y[m].mean()
        rel += w * (pbar - ybar) ** 2
        res += w * (ybar - base) ** 2
    return rel, res, base * (1 - base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=2024)
    ap.add_argument("--apply", type=int, default=2025)
    args = ap.parse_args()

    paths = {s: os.path.join(config.PROCESSED_DIR, f"sim_preds_{s}.csv")
             for s in (args.fit, args.apply)}
    for s, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nrun: python evaluate_sim.py --season {s} --sims 1000")

    tr = pd.read_csv(paths[args.fit])
    te = pd.read_csv(paths[args.apply])
    print(f"fit on {args.fit} ({len(tr):,} games), "
          f"apply to {args.apply} ({len(te):,} games)")

    a, b = fit_platt(tr["p_home"].to_numpy(), tr["home_won"].to_numpy())
    print(f"\nPlatt parameters: a={a:.4f}  b={b:+.4f}")
    print("  a below 1 means the simulator was overconfident and its "
          "probabilities get pulled toward the base rate;")
    print("  a above 1 means it was underconfident and they get pushed out.")

    p_raw = te["p_home"].to_numpy()
    y = te["home_won"].to_numpy()
    p_cal = apply_platt(p_raw, a, b)

    print(f"\n{'':<22} {'log loss':>10} {'Brier':>10} {'accuracy':>10} {'AUC':>8}")
    for label, p in [("simulator raw", p_raw), ("simulator calibrated", p_cal)]:
        ll, br, acc = metrics(p, y)
        print(f"{label:<22} {ll:>10.5f} {br:>10.5f} {acc * 100:>9.2f}% "
              f"{auc(p, y):>8.5f}")

    print("\nreference points on the same games:")
    print(f"{'always home':<22} {'0.68989':>10} {'0.24837':>10} {'54.03%':>10}")
    print(f"{'talent logistic':<22} {'0.68065':>10} {'0.24391':>10} {'54.61%':>10}")
    print(f"{'Elo (record only)':<22} {'0.67936':>10} {'0.24332':>10} {'56.42%':>10}")
    print(f"{'Elo + talent':<22} {'0.67789':>10} {'0.24261':>10} {'55.93%':>10}")

    print("\nMurphy decomposition (Brier = reliability - resolution + uncertainty):")
    for label, p in [("raw", p_raw), ("calibrated", p_cal)]:
        rel, res, unc = murphy(p, y)
        print(f"  {label:<12} reliability {rel:.5f}   resolution {res:.5f}   "
              f"uncertainty {unc:.5f}")
    print("  Reliability is miscalibration and is what Platt scaling removes.")
    print("  Resolution is discrimination -- it is what a better model would "
          "raise, and no output transform can.")

    print("\ncalibration after scaling:")
    bins = [0, .35, .40, .45, .50, .55, .60, .65, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_cal >= lo) & (p_cal < hi)
        if m.sum() < 10:
            continue
        print(f"  {lo:.2f}-{hi:.2f}: predicted {p_cal[m].mean() * 100:5.1f}%  "
              f"actual {y[m].mean() * 100:5.1f}%  (n={m.sum():,})")


if __name__ == "__main__":
    main()
