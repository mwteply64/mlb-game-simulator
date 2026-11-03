"""Arm-slot archetype, tested properly: PA-level, out of sample, ten reps.

The per-swing version survived its permutation control -- 71% of the gain died
when batters were reassigned -- which no other feature in this project has
managed. But that test had two weaknesses worth fixing before anyone builds on
it:

* **It measured whiffs per swing, not strikeouts per plate appearance.** A prop
  is priced in strikeouts, and a whiff effect has to survive two translations to
  get there.
* **It fitted the batter rates and evaluated them on the same rows.** A hitter's
  own outcomes were inside the rates used to predict those outcomes, which
  inflates any batter-specific effect -- exactly the shape of effect being
  claimed. Here the rates come from strictly prior seasons and the test runs on
  a season they never saw.

Note what the tilt actually is. Arm angle is close to a constant for a given
pitcher, so a pitcher sits in one bucket and his own bucket-relative rate is 1
by construction. The pitcher side therefore contributes nothing, and the feature
reduces to: *this hitter's strikeout rate against this kind of arm, relative to
his own overall rate*. That is the archetype claim in its purest form, and
shuffling batters is exactly the right way to break it.

    python armslot_test.py --fit 2023 2024 --test 2025 --reps 10
"""

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om

K_SHRINK = 250.0        # PA of league prior on a batter's per-bucket rate


def pitcher_arm_angle():
    """Mean arm angle per pitcher-season, from the raw pitch files."""
    rows = []
    for f in sorted(glob.glob(os.path.join(config.RAW_DIR, "statcast_*.parquet"))):
        d = pd.read_parquet(f)
        if "arm_angle" not in d.columns:
            continue
        s = int(os.path.basename(f).split("_")[1])
        g = d.groupby("pitcher")["arm_angle"].agg(["mean", "size"]).reset_index()
        g["season"] = s
        rows.append(g)
    a = pd.concat(rows, ignore_index=True)
    a = (a.groupby(["pitcher", "season"])
         .apply(lambda x: np.average(x["mean"], weights=x["size"]))
         .rename("arm_angle").reset_index())
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--test", type=int, default=2025)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--buckets", type=int, default=3)
    args = ap.parse_args()

    pa = om.load_pa()
    arm = pitcher_arm_angle()
    d = pa[pa["is_pa"] & pa["outcome"].isin(config.OUTCOMES)][
        ["season", "batter", "pitcher", "outcome"]].copy()
    d = d.merge(arm, on=["pitcher", "season"], how="inner").dropna(
        subset=["arm_angle"])
    d["is_k"] = (d["outcome"] == "K").astype(float)

    fit = d[d["season"].isin(args.fit)]
    test = d[d["season"] == args.test]
    if not len(fit) or not len(test):
        raise SystemExit("no data for the requested seasons")

    edges = np.quantile(fit["arm_angle"], np.linspace(0, 1, args.buckets + 1)[1:-1])
    for f in (fit, test):
        f["bucket"] = np.searchsorted(edges, f["arm_angle"].to_numpy())

    lg = fit.groupby("bucket")["is_k"].mean()
    lg_all = float(fit["is_k"].mean())
    print(f"arm-slot buckets fitted on {args.fit}, tested on {args.test}")
    print("  league K rate by bucket: " +
          "  ".join(f"b{int(k)} {v * 100:.2f}%" for k, v in lg.items()))
    print(f"  (flat buckets mean no main effect to leak)\n")

    # Batter rates from the fit seasons only.
    g = fit.groupby(["batter", "bucket"])["is_k"].agg(["sum", "size"])
    o = fit.groupby("batter")["is_k"].agg(["sum", "size"])
    b_all = ((o["sum"] + K_SHRINK * lg_all) / (o["size"] + K_SHRINK)).to_dict()
    b_bkt = {}
    for (pid, b), r in g.iterrows():
        v = b_bkt.setdefault(int(pid), np.full(args.buckets, np.nan))
        v[int(b)] = (r["sum"] + K_SHRINK * lg[b]) / (r["size"] + K_SHRINK)

    def tilt_for(batters, buckets):
        out = np.zeros(len(batters))
        for i, (bt, bk) in enumerate(zip(batters, buckets)):
            v = b_bkt.get(int(bt))
            b0 = b_all.get(int(bt))
            if v is None or not b0 or np.isnan(v[int(bk)]):
                continue
            out[i] = np.log(max(v[int(bk)] / b0, 1e-6))
        return out

    # Baseline: the PA model's own strikeout probability on the test season.
    tables = om.build_count_tables(pa)
    model = om.OutcomeModel(tables, args.test)
    ki = config.OUTCOMES.index("K")
    t = test.copy()
    tp = pa[pa["is_pa"] & (pa["season"] == args.test)][
        ["batter", "pitcher", "stand", "p_throws"]]
    t = t.join(tp[["stand", "p_throws"]])
    cache, base = {}, np.empty(len(t))
    for i, (b, p, st, ph) in enumerate(zip(t["batter"], t["pitcher"],
                                           t["stand"].fillna("R"),
                                           t["p_throws"].fillna("R"))):
        key = (b, p, st, ph)
        v = cache.get(key)
        if v is None:
            v = float(model.matchup(b, p, st, ph)[ki])
            cache[key] = v
        base[i] = v
    z0 = np.log(np.clip(base, 1e-6, 1 - 1e-6) /
                np.clip(1 - base, 1e-6, 1 - 1e-6))
    y = t["is_k"].to_numpy()

    def ll_with(tv):
        beta = 0.0
        for _ in range(60):
            p = 1.0 / (1.0 + np.exp(-(z0 + beta * tv)))
            gr = np.mean((y - p) * tv)
            h = np.mean(p * (1 - p) * tv * tv) + 1e-9
            step = gr / h
            beta += step
            if abs(step) < 1e-10:
                break
        p = np.clip(1.0 / (1.0 + np.exp(-(z0 + beta * tv))), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), beta

    null = float(-np.mean(y * np.log(np.clip(base, 1e-6, 1 - 1e-6)) +
                          (1 - y) * np.log(np.clip(1 - base, 1e-6, 1 - 1e-6))))
    real_t = tilt_for(t["batter"].to_numpy(), t["bucket"].to_numpy())
    real, beta = ll_with(real_t)
    print(f"{len(t):,} held-out plate appearances")
    print(f"  baseline log loss     {null:.6f}")
    print(f"  + arm-slot tilt       {real:.6f}   ({real - null:+.6f})"
          f"   beta {beta:+.4f}")
    print(f"  tilt sd {real_t.std():.4f}, nonzero {100 * (real_t != 0).mean():.0f}%")

    rng = np.random.default_rng(23)
    gains = []
    for _ in range(args.reps):
        perm = rng.permutation(len(t))
        g_, _ = ll_with(tilt_for(t["batter"].to_numpy()[perm],
                                 t["bucket"].to_numpy()))
        gains.append(g_ - null)
    gains = np.array(gains)
    print(f"\n  batters shuffled ({args.reps} reps): mean {gains.mean():+.6f}"
          f"  sd {gains.std():.6f}")
    print(f"  real {real - null:+.6f}")
    surv = gains.mean() / (real - null) * 100 if (real - null) else float("nan")
    print(f"  share of gain surviving the shuffle: {surv:.0f}%")
    if real - null >= -1e-9:
        print("  -> no gain at all out of sample")
    elif surv > 50:
        print("  -> mostly survives shuffling: not a batter-specific effect")
    else:
        print("  -> genuine batter x arm-slot interaction")


if __name__ == "__main__":
    main()
