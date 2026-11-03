"""Correct the strikeout slope, not just its level.

`diagnose_k.py` found that the PA model over-predicts strikeouts in every
season, and that the error grows with the prediction: the lowest bucket is
under-predicted by 2.8% while the highest is over by 5.0%. That is not a level
bias, it is a slope bias -- the odds-ratio multiplication spreads strikeout
probability too wide, so a high-K pitcher against a high-K lineup compounds past
what actually happens.

The existing `sim_calibration.json` cannot fix this. It is a single multiplier
per outcome, which shifts the level and leaves the slope untouched.

So this fits the standard two-parameter recalibration in log-odds space,

    logit(p') = a + b * logit(p)

where b < 1 pulls extreme predictions back toward the league. Fitted on one
season, verified on seasons it never saw. The pitchers whose strikeout props are
worth pricing all live in the tail where b matters most, so this is aimed
directly at the only part of the project with plausible value.

    python fit_k_calibration.py --fit 2024 --verify 2025 2026
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om
from diagnose_k_start import pa_predictions

# The props pipeline refreshes its model monthly with season-to-date counts,
# which already removes part of the bias (2025: +4.0% -> +3.1%). Fitting on the
# strictly-prior-seasons configuration would produce a constant too large for
# the pipeline it is meant to correct, so this matches PropSimulator exactly.
SEASON_TO_DATE = True


def predicted_k(pa, tables, season):
    return pa_predictions(pa, tables, season, season_to_date=SEASON_TO_DATE)

PATH = os.path.join(config.PROCESSED_DIR, "k_calibration.json")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_platt(p, y, iters=200):
    """Newton on (a, b) for logit(p') = a + b*logit(p)."""
    z = logit(p)
    a, b = 0.0, 1.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(a + b * z)))
        w = np.maximum(q * (1 - q), 1e-12)
        r = y - q
        # 2x2 normal equations with design [1, z]
        g = np.array([r.sum(), (r * z).sum()])
        h = np.array([[w.sum(), (w * z).sum()],
                      [(w * z).sum(), (w * z * z).sum()]])
        step = np.linalg.solve(h + np.eye(2) * 1e-9, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def apply(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


def report(d, a, b, label):
    y = d["is_k"].to_numpy()
    p = d["p_k"].to_numpy()
    q = apply(p, a, b)
    ll0, ll1 = log_loss(p, y), log_loss(q, y)
    print(f"\n=== {label} ({len(d):,} PAs) ===")
    print(f"  rate     before {p.mean() * 100:5.2f}%   after {q.mean() * 100:5.2f}%"
          f"   actual {y.mean() * 100:5.2f}%")
    print(f"  log loss before {ll0:.6f}   after {ll1:.6f}   ({ll1 - ll0:+.6f})")

    dd = d.assign(q=q)
    dd["bucket"] = pd.cut(dd["p_k"], [0, .15, .20, .25, .30, 1.0])
    g = dd.groupby("bucket").agg(pred=("p_k", "mean"), post=("q", "mean"),
                                 act=("is_k", "mean"), n=("is_k", "size"))
    for bk, r in g.iterrows():
        if r["n"] < 500:
            continue
        print(f"    P(K) {str(bk):<13} before {r['pred'] * 100:5.2f}%  "
              f"after {r['post'] * 100:5.2f}%  actual {r['act'] * 100:5.2f}%  "
              f"({(r['post'] - r['act']) / r['act'] * 100:+.1f}% vs "
              f"{(r['pred'] - r['act']) / r['act'] * 100:+.1f}%)")

    if "is_starter" in d.columns:
        st = dd[dd["is_starter"]]
        if len(st):
            bias0 = (st["p_k"].mean() - st["is_k"].mean()) * 24
            bias1 = (st["q"].mean() - st["is_k"].mean()) * 24
            print(f"  starter 24-BF strikeout bias: {bias0:+.2f} -> {bias1:+.2f} K")
    return ll1 - ll0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=2024)
    ap.add_argument("--verify", type=int, nargs="*", default=[2025, 2026])
    args = ap.parse_args()

    pa = om.load_pa()
    tables = om.build_count_tables(pa)

    dfit = predicted_k(pa, tables, args.fit)
    a, b = fit_platt(dfit["p_k"].to_numpy(), dfit["is_k"].to_numpy())
    print(f"fitted on {args.fit}:  a = {a:+.5f}   b = {b:.5f}")
    print(f"  b < 1 means the model's strikeout spread is too wide; "
          f"this pulls it in by {(1 - b) * 100:.1f}%")
    report(dfit, a, b, f"{args.fit} (in-sample)")

    for v in args.verify:
        dv = predicted_k(pa, tables, v)
        if len(dv):
            report(dv, a, b, f"{v} (held out)")

    with open(PATH, "w") as fh:
        json.dump({"season_fit": args.fit, "a": a, "b": b,
                   "note": "logit(p_K') = a + b*logit(p_K); "
                           "corrects slope, not just level"}, fh, indent=2)
    print(f"\nwrote {PATH}")


if __name__ == "__main__":
    main()
