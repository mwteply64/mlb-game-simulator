"""Does *how* a pitcher gets strikeouts add anything to *how many* he gets?

The PA model multiplies a batter's overall rates by a pitcher's overall rates
through a league denominator. Two pitchers with the same strikeout rate are
identical to it, even if one is slider-heavy and the other lives off a
changeup. This asks whether that distinction predicts anything once both
players' overall ability is already accounted for.

The feature is deliberately a single number, not a pile of them. For a matchup
it compares the whiff rate the arsenal implies against the whiff rate the two
players' overall numbers imply:

    E  = sum_g share_g(pitcher) * batter_whiff_g * pitcher_whiff_g / league_g
    E0 =                          batter_whiff   * pitcher_whiff   / league

    tilt = log(E / E0)

`tilt` is positive when a pitcher's specific mix attacks a specific hitter's
specific hole, and zero when the arsenal tells you nothing the overall rates
did not. Because both players' general ability cancels in the ratio, a
coefficient on `tilt` is a clean test of the arsenal hypothesis rather than a
restatement of "good pitchers strike people out."

Whiff-per-swing is used rather than outcome rates by finishing pitch group.
Finishing-group rates are confounded: breaking balls end PAs at twice the
strikeout rate of sinkers largely because they are thrown in two-strike counts.
Whiffs are measured on every pitch and survive that objection.

    python arsenal.py --fit 2024 --test 2025
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om
from build_pitch_tables import GROUPS

P = config.PROCESSED_DIR
K_IDX = config.OUTCOMES.index("K")

# Shrinkage constants, in swings. A hitter takes ~450 swings a season against
# fastballs but only ~120 against offspeed, so the rarer groups need to regress
# harder or the tilt is dominated by noise in the small cells.
K_WHIFF = 350.0
K_MIX = 400.0


def _asof(tbl, season, decay=0.7):
    """Prior seasons with exponential decay, matching the PA model's convention.

    Strictly prior seasons only. The season being predicted is excluded rather
    than partially included: a season-to-date term would need the same
    game-by-game bookkeeping the PA model does, and the arsenal is the slowest
    moving thing in this project -- a pitcher's mix in April predicts his mix in
    September far better than his ERA does.
    """
    h = tbl[tbl["season"] < season].copy()
    if not len(h):
        return h
    h["w"] = decay ** (season - h["season"] - 1)
    return h


def whiff_rates(season, role):
    """Per-group whiff-per-swing by player, shrunk toward that group's league rate."""
    t = _asof(pd.read_parquet(os.path.join(P, f"pitch_{role}_whiff.parquet")), season)
    if not len(t):
        return {}, {}, np.zeros(len(GROUPS))

    t["swings"] = t["swings"] * t["w"]
    t["whiffs"] = t["whiffs"] * t["w"]
    g = t.groupby(["player_id", "group"], as_index=False)[["swings", "whiffs"]].sum()

    lg = t.groupby("group")[["swings", "whiffs"]].sum()
    lg_rate = (lg["whiffs"] / lg["swings"].clip(lower=1)).reindex(GROUPS).fillna(0.2)
    lg_overall = float(t["whiffs"].sum() / max(t["swings"].sum(), 1))

    gi = {name: i for i, name in enumerate(GROUPS)}
    g = g[g["group"].isin(gi)]
    idx = g["group"].map(gi).to_numpy()
    pid = g["player_id"].to_numpy()
    lgv = lg_rate.to_numpy()

    by_group, overall_sw, overall_wh = {}, {}, {}
    for p, i, sw, wh in zip(pid, idx, g["swings"].to_numpy(), g["whiffs"].to_numpy()):
        v = by_group.setdefault(int(p), lgv.copy())
        v[i] = (wh + K_WHIFF * lgv[i]) / (sw + K_WHIFF)
        overall_sw[int(p)] = overall_sw.get(int(p), 0.0) + sw
        overall_wh[int(p)] = overall_wh.get(int(p), 0.0) + wh

    overall = {p: (overall_wh[p] + K_WHIFF * lg_overall) / (overall_sw[p] + K_WHIFF)
               for p in overall_sw}
    return by_group, overall, lgv, lg_overall


def arsenal_mix(season):
    """Share of pitches by group, shrunk toward the league mix."""
    t = _asof(pd.read_parquet(os.path.join(P, "pitch_pitcher_mix.parquet")), season)
    if not len(t):
        return {}, np.ones(len(GROUPS)) / len(GROUPS)

    t["n"] = t["n_pitches"] * t["w"]
    gi = {name: i for i, name in enumerate(GROUPS)}
    t = t[t["group"].isin(gi)]
    lg = t.groupby("group")["n"].sum().reindex(GROUPS).fillna(0.0).to_numpy()
    lg_mix = lg / max(lg.sum(), 1)

    counts = {}
    for p, grp, n in zip(t["pitcher"].to_numpy(), t["group"].map(gi).to_numpy(),
                         t["n"].to_numpy()):
        v = counts.setdefault(int(p), np.zeros(len(GROUPS)))
        v[grp] += n

    mix = {}
    for p, v in counts.items():
        m = (v + K_MIX * lg_mix) / (v.sum() + K_MIX)
        mix[p] = m / m.sum()
    return mix, lg_mix


class ArsenalTilt:
    def __init__(self, season):
        self.b_grp, self.b_all, self.lg_grp, self.lg_all = whiff_rates(season, "batter")
        self.p_grp, self.p_all, _, _ = whiff_rates(season, "pitcher")
        self.mix, self.lg_mix = arsenal_mix(season)

    def tilt(self, batter_id, pitcher_id):
        """Interaction only, with both main effects divided out.

        An earlier version compared a mix-weighted per-group expectation
        against an unweighted overall one. Those are not on the same scale, and
        the mismatch tracks arsenal *type*: a sinker-heavy pitcher concentrates
        his mix in the lowest-whiff group, which inflated his score even though
        he strikes out fewer batters. The feature ended up anti-correlated with
        strikeouts and merely restated what the PA model already knew.

        Here each player's per-group rate is expressed relative to his own
        overall rate, so a player with no group-specific profile contributes
        exactly 1 and the tilt is 0. What survives is only whether this
        pitcher's mix happens to point at this hitter's specific hole.
        """
        b = self.b_grp.get(int(batter_id))
        p = self.p_grp.get(int(pitcher_id))
        m = self.mix.get(int(pitcher_id))
        if b is None or p is None or m is None:
            return 0.0
        b0 = self.b_all.get(int(batter_id), self.lg_all)
        p0 = self.p_all.get(int(pitcher_id), self.lg_all)
        if b0 <= 0 or p0 <= 0:
            return 0.0

        # Weight by the whiffs each group is expected to contribute, not by raw
        # pitch share: a group thrown often but rarely swung through should not
        # dominate the average.
        w = m * self.lg_grp
        s = w.sum()
        if s <= 0:
            return 0.0
        w = w / s

        rel = float(np.sum(w * (b / b0) * (p / p0)))
        return float(np.log(rel)) if rel > 0 else 0.0


def pa_frame(pa, season, tilt_obj, model):
    """Baseline P(K) from the PA model, plus the arsenal tilt, for every PA."""
    d = pa[pa["is_pa"] & (pa["season"] == season)]
    d = d[["batter", "pitcher", "stand", "p_throws", "outcome",
           "n_thruorder_pitcher"]].copy()

    # matchup() is per-PA python; cache on the matchup key, which collapses
    # ~180k PAs into ~40k distinct combinations.
    cache = {}
    pk, tl = np.empty(len(d)), np.empty(len(d))
    for i, (b, p, st, ph, tto) in enumerate(zip(
            d["batter"].to_numpy(), d["pitcher"].to_numpy(),
            d["stand"].to_numpy(), d["p_throws"].to_numpy(),
            d["n_thruorder_pitcher"].fillna(1).to_numpy())):
        key = (b, p, st, ph, min(int(tto), 4))
        got = cache.get(key)
        if got is None:
            got = (float(model.matchup(b, p, st, ph, tto=int(tto))[K_IDX]),
                   tilt_obj.tilt(b, p))
            cache[key] = got
        pk[i], tl[i] = got

    d["p_k"] = pk
    d["tilt"] = tl
    d["is_k"] = (d["outcome"] == "K").astype(float)
    return d


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_beta(d):
    """One-parameter logistic refit: logit(P(K)) + beta * tilt."""
    z, t, y = logit(d["p_k"].to_numpy()), d["tilt"].to_numpy(), d["is_k"].to_numpy()
    beta = 0.0
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-(z + beta * t)))
        grad = np.mean((y - p) * t)
        hess = np.mean(p * (1 - p) * t * t) + 1e-9
        step = grad / hess
        beta += step
        if abs(step) < 1e-9:
            break
    return float(beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=2024)
    ap.add_argument("--test", type=int, default=2025)
    args = ap.parse_args()

    pa = om.load_pa()
    tables = om.build_count_tables(pa)

    out, tilt_objs = {}, {}
    for season in (args.fit, args.test):
        model = om.OutcomeModel(tables, season)
        tilt_objs[season] = ArsenalTilt(season)
        d = pa_frame(pa, season, tilt_objs[season], model)
        d["batter"] = pa[pa["is_pa"] & (pa["season"] == season)]["batter"].to_numpy()
        d["pitcher"] = pa[pa["is_pa"] & (pa["season"] == season)]["pitcher"].to_numpy()
        out[season] = d
        print(f"  {season}: {len(d):,} PAs, tilt sd {d['tilt'].std():.4f}, "
              f"nonzero {100 * (d['tilt'] != 0).mean():.1f}%")

    beta = fit_beta(out[args.fit])
    print(f"\nfitted on {args.fit}: beta = {beta:+.4f}")

    d = out[args.test]
    y = d["is_k"].to_numpy()
    base = d["p_k"].to_numpy()
    adj = 1.0 / (1.0 + np.exp(-(logit(base) + beta * d["tilt"].to_numpy())))

    ll0, ll1 = log_loss(base, y), log_loss(adj, y)
    print(f"\n=== held out on {args.test} ({len(d):,} PAs) ===")
    print(f"  baseline PA model   log loss {ll0:.6f}")
    print(f"  + arsenal tilt      log loss {ll1:.6f}   ({ll1 - ll0:+.6f})")
    print(f"  actual K rate {y.mean() * 100:.2f}%   predicted {base.mean() * 100:.2f}%")

    # Per-PA gain is tiny by construction; a strikeout prop is a sum over ~24
    # batters faced, so the figure that matters is the change in a start's
    # projected total, not the change in one PA.
    se = (np.std(-(y * np.log(np.clip(adj, 1e-6, 1)) +
                   (1 - y) * np.log(np.clip(1 - adj, 1e-6, 1))) +
                 (y * np.log(np.clip(base, 1e-6, 1)) +
                  (1 - y) * np.log(np.clip(1 - base, 1e-6, 1)))) / np.sqrt(len(y)))
    print(f"  paired SE {se:.6f}  ->  {abs(ll1 - ll0) / max(se, 1e-12):.1f} sigma")

    # -- the control that decides what this is ------------------------------
    # Reassign batters at random while leaving pitchers untouched. That erases
    # any genuine batter-specific arsenal fit but preserves arsenal *type*
    # exactly. A real interaction must die here. Anything that survives was a
    # pitcher main effect wearing an interaction's clothes.
    print("\n  === permutation control: batters shuffled within season ===")
    rng = np.random.default_rng(17)
    gains = []
    for rep in range(3):
        for season in (args.fit, args.test):
            src = out[season]
            perm = rng.permutation(len(src))
            tilt_obj = tilt_objs[season]
            bat_shuf = src["batter"].to_numpy()[perm]
            tl = np.array([tilt_obj.tilt(b, p) for b, p in
                           zip(bat_shuf, src["pitcher"].to_numpy())])
            src = src.assign(tilt=tl)
            out.setdefault("_ctl", {})[season] = src
        b_ctl = fit_beta(out["_ctl"][args.fit])
        dc = out["_ctl"][args.test]
        adj_c = 1.0 / (1.0 + np.exp(-(logit(dc["p_k"].to_numpy())
                                      + b_ctl * dc["tilt"].to_numpy())))
        g = log_loss(adj_c, dc["is_k"].to_numpy()) - ll0
        gains.append(g)
        print(f"    rep {rep + 1}: beta {b_ctl:+.4f}   held-out {g:+.6f}")
    print(f"    mean shuffled gain {np.mean(gains):+.6f}   "
          f"vs real {ll1 - ll0:+.6f}")
    if np.mean(gains) <= (ll1 - ll0) * 0.5:
        print("    -> most of the gain survives shuffling: this is a pitcher "
              "main effect, not arsenal fit")
    else:
        print("    -> gain collapses when batters are shuffled: genuine "
              "batter-arsenal interaction")

    q = d["tilt"].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    print("\n  tilt distribution (log scale): "
          + "  ".join(f"p{int(k * 100)} {v:+.3f}" for k, v in q.items()))
    hi = d[d["tilt"] > q[0.75]]
    lo = d[d["tilt"] < q[0.25]]
    print(f"  actual K rate in top quartile of tilt  {hi['is_k'].mean() * 100:.2f}%"
          f"   (model says {hi['p_k'].mean() * 100:.2f}%)")
    print(f"  actual K rate in bottom quartile       {lo['is_k'].mean() * 100:.2f}%"
          f"   (model says {lo['p_k'].mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
