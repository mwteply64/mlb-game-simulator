"""Does pitch physics beat past results, and do hitters have archetype holes?

Three questions, all tested the same way the arsenal hypothesis was -- against a
control that destroys the effect being claimed while leaving everything else in
place. The arsenal feature looked like 5.6 sigma until a permutation control
showed 57% of it survived shuffling the batters, so it was never shipped.

1. **Stuff+.** Fit whiff-per-swing from pitch physics alone -- velocity, both
   movement planes, spin rate, arm angle, plate location -- with no reference to
   who threw it or what happened to him. Aggregate to a pitcher-season. Then ask
   the question that matters: does last season's *stuff* predict this season's
   strikeout rate beyond last season's *strikeout rate*? That is the claim the
   public metrics make, and it is the one worth testing, because our model
   currently projects a pitcher purely from his own shrunk history.

2. **Velocity archetype.** Do hitters have velocity-specific holes -- beyond
   being good or bad hitters -- that a fast pitcher can exploit?

3. **Arm-slot archetype.** The same for release angle.

Both archetype tests reuse the arsenal construction: express each side's
per-bucket rate relative to its own overall rate, so a player with no
bucket-specific profile contributes exactly 1 and the interaction is 0. Then
shuffle the batters and see how much of the gain survives.

    python stuff_test.py
"""

import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
PHYS = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "arm_angle",
        "plate_x", "plate_z"]


def load(seasons, max_per_season=900_000):
    frames = []
    for s in seasons:
        got = []
        for f in sorted(glob.glob(os.path.join(config.RAW_DIR,
                                               f"statcast_{s}_*.parquet"))):
            cols = ["pitcher", "batter", "description", "stand", "p_throws",
                    "pitch_type"] + PHYS
            d = pd.read_parquet(f)
            got.append(d[[c for c in cols if c in d.columns]])
        if not got:
            continue
        d = pd.concat(got, ignore_index=True)
        d["season"] = s
        if len(d) > max_per_season:
            d = d.sample(max_per_season, random_state=7)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["swing"] = d["description"].isin(SWING)
    d["whiff"] = d["description"].isin(WHIFF)
    return d.dropna(subset=[c for c in PHYS if c in d.columns])


def fit_logistic(X, y, iters=40, l2=1.0):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p) - l2 * np.r_[0.0, w[1:]]
        s = np.maximum(p * (1 - p), 1e-6)
        H = (X * s[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def stuff_test(d):
    print("=" * 66)
    print("1. STUFF+  --  does physics predict next season beyond past results?")
    print("=" * 66)
    sw = d[d["swing"]].copy()
    feats = [c for c in PHYS if c in sw.columns]
    # Physical features only. No pitcher identity, no outcome history: if the
    # pitcher's own id leaked in, this would just relearn his strikeout rate.
    X = sw[feats].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.column_stack([Xs, Xs[:, 0] ** 2, np.abs(Xs[:, 5]), np.abs(Xs[:, 6])])

    tr = sw["season"] <= 2024
    w = fit_logistic(Xs[tr.to_numpy()], sw.loc[tr, "whiff"].to_numpy(float))
    pred = 1.0 / (1.0 + np.exp(-np.clip(
        np.column_stack([np.ones(len(Xs)), Xs]) @ w, -30, 30)))
    sw["stuff"] = pred
    print(f"   fitted on {int(tr.sum()):,} swings (<=2024), "
          f"applied to {len(sw):,}")

    # Pitcher-season aggregates: stuff, and actual K rate per batter faced.
    ps = sw.groupby(["pitcher", "season"]).agg(
        stuff=("stuff", "mean"), n=("stuff", "size")).reset_index()
    ps = ps[ps["n"] >= 200]

    kr = (d.groupby(["pitcher", "season"])
          .agg(k=("description", lambda s: s.isin(
              ["swinging_strike", "swinging_strike_blocked",
               "called_strike"]).sum()), n=("description", "size")).reset_index())
    kr["k_rate"] = kr["k"] / kr["n"].clip(lower=1)
    ps = ps.merge(kr[["pitcher", "season", "k_rate"]], on=["pitcher", "season"])

    prev = ps.copy(); prev["season"] += 1
    j = ps.merge(prev, on=["pitcher", "season"], suffixes=("", "_prev"))
    if len(j) < 50:
        print("   too few pitcher-seasons to test"); return
    print(f"   {len(j)} pitcher-season pairs\n")

    a = np.corrcoef(j["k_rate_prev"], j["k_rate"])[0, 1]
    b = np.corrcoef(j["stuff_prev"], j["k_rate"])[0, 1]
    print(f"   correlation with THIS season's whiff/called-strike rate:")
    print(f"     last season's rate  {a:+.4f}")
    print(f"     last season's stuff {b:+.4f}")

    X2 = np.column_stack([np.ones(len(j)), j["k_rate_prev"], j["stuff_prev"]])
    wt = np.linalg.lstsq(X2, j["k_rate"], rcond=None)[0]
    r1 = a ** 2
    r2 = np.corrcoef(X2 @ wt, j["k_rate"])[0, 1] ** 2
    print(f"\n   R2: past rate alone {r1:.4f}   + stuff {r2:.4f}   "
          f"(gain {r2 - r1:+.4f})")
    print(f"   weights: past rate {wt[1]:+.4f}   stuff {wt[2]:+.4f}")
    if r2 - r1 < 0.005:
        print("   -> stuff adds essentially nothing beyond past results here")
    else:
        print("   -> stuff carries information past results do not")


def archetype_test(d, col, label, n_buckets=3):
    print("\n" + "=" * 66)
    print(f"{label.upper()}  --  do hitters have {label}-specific holes?")
    print("=" * 66)
    sw = d[d["swing"] & d[col].notna()].copy()
    if not len(sw):
        print("   column unavailable"); return

    edges = np.quantile(sw[col], [1 / n_buckets, 2 / n_buckets])
    sw["bucket"] = np.searchsorted(edges, sw[col].to_numpy())

    lg = sw.groupby("bucket")["whiff"].mean()
    print("   league whiff/swing by bucket: " +
          "  ".join(f"b{int(k)} {v * 100:.1f}%" for k, v in lg.items()))

    K = 300.0
    def rates(key):
        g = sw.groupby([key, "bucket"])["whiff"].agg(["sum", "size"])
        o = sw.groupby(key)["whiff"].agg(["sum", "size"])
        overall = ((o["sum"] + K * sw["whiff"].mean()) / (o["size"] + K))
        out = {}
        for (pid, b), r in g.iterrows():
            v = out.setdefault(int(pid), np.full(n_buckets, np.nan))
            v[int(b)] = (r["sum"] + K * lg[b]) / (r["size"] + K)
        return out, overall.to_dict()

    b_r, b_o = rates("batter")
    p_r, p_o = rates("pitcher")

    def tilt(bat, pit, bkt):
        br, pr = b_r.get(int(bat)), p_r.get(int(pit))
        if br is None or pr is None:
            return 0.0
        b0, p0 = b_o.get(int(bat)), p_o.get(int(pit))
        if not b0 or not p0 or np.isnan(br[bkt]) or np.isnan(pr[bkt]):
            return 0.0
        return float(np.log(max((br[bkt] / b0) * (pr[bkt] / p0), 1e-6)))

    s = sw.sample(min(len(sw), 260_000), random_state=3)
    t = np.array([tilt(b, p, k) for b, p, k in
                  zip(s["batter"], s["pitcher"], s["bucket"])])
    y = s["whiff"].to_numpy(float)
    base = np.log(np.clip(np.array([b_o.get(int(b), 0.25) for b in s["batter"]])
                          * np.array([p_o.get(int(p), 0.25) for p in s["pitcher"]])
                          / max(sw["whiff"].mean(), 1e-9), 1e-6, 1 - 1e-6))
    z0 = np.log(np.clip(1 / (1 + np.exp(-base)), 1e-6, 1 - 1e-6))

    def ll(t_vec):
        w = fit_logistic(np.column_stack([z0, t_vec]), y)
        p = 1 / (1 + np.exp(-np.clip(
            np.column_stack([np.ones(len(y)), z0, t_vec]) @ w, -30, 30)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), w[2]

    real, coef = ll(t)
    null, _ = ll(np.zeros_like(t))
    print(f"   real interaction: log loss {real:.6f} vs {null:.6f} "
          f"({real - null:+.6f}), coefficient {coef:+.4f}")

    rng = np.random.default_rng(11)
    gains = []
    for _ in range(3):
        perm = rng.permutation(len(s))
        ts = np.array([tilt(b, p, k) for b, p, k in
                       zip(s["batter"].to_numpy()[perm], s["pitcher"],
                           s["bucket"])])
        g, _ = ll(ts)
        gains.append(g - null)
    m = float(np.mean(gains))
    print(f"   batters shuffled: {m:+.6f} (mean of 3)")
    if real - null >= 0 or m <= (real - null) * 0.5:
        print("   -> no real batter-specific effect: it is a main effect or noise")
    else:
        print("   -> survives the control; genuine archetype interaction")


def main():
    d = load([2023, 2024, 2025])
    print(f"loaded {len(d):,} pitches with full physics\n")
    stuff_test(d)
    archetype_test(d, "release_speed", "velocity")
    archetype_test(d, "arm_angle", "arm slot")


if __name__ == "__main__":
    main()
