"""Held-out evaluation of the plate appearance outcome model.

The gate is multiclass log loss on plate appearances the model never saw, with
every component added one at a time so each one has to earn its place. Two
modes:

    --mode prior   project a whole season from strictly prior seasons only
    --mode asof    walk forward through the season by month, folding in
                   season-to-date data as it would actually be available

`asof` is the realistic number. `prior` is the conservative floor and runs much
faster, so it is the default for quick iteration.

    python evaluate_outcome_model.py --holdout 2025
    python evaluate_outcome_model.py --holdout 2025 --mode asof
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)


def log_loss(probs, y_idx, eps=1e-12):
    return float(-np.mean(np.log(np.maximum(probs[np.arange(len(y_idx)), y_idx], eps))))


def brier(probs, y_idx):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


class VectorCache:
    """Precomputes one vector per (player, opposing hand) so the per-plate-
    appearance loop is a dict lookup instead of a platoon recomputation."""

    def __init__(self, model):
        self.m = model
        self._bat = {}
        self._pit = {}

    def batter(self, pid, stand, hand):
        key = (pid, stand, hand)
        v = self._bat.get(key)
        if v is None:
            v = self.m.batter_vector(pid, stand, hand)
            self._bat[key] = v
        return v

    def pitcher(self, pid, hand, stand):
        key = (pid, hand, stand)
        v = self._pit.get(key)
        if v is None:
            v = self.m.pitcher_vector(pid, hand, stand)
            self._pit[key] = v
        return v


def predict_block(model, block, use_platoon=True, use_tto=False,
                  park_factors=None, side="both", calibration=None):
    """Vectorized-enough prediction over a block of plate appearances."""
    cache = VectorCache(model)
    n = len(block)
    lg = model.league

    B = np.empty((n, N_OUT))
    P = np.empty((n, N_OUT))

    bat_ids = block["batter"].to_numpy()
    pit_ids = block["pitcher"].to_numpy()
    stands = block["stand"].to_numpy()
    hands = block["p_throws"].to_numpy()

    for i in range(n):
        s, h = stands[i], hands[i]
        if use_platoon:
            B[i] = cache.batter(int(bat_ids[i]), s, h)
            P[i] = cache.pitcher(int(pit_ids[i]), h, s)
        else:
            B[i] = model.bat_rates.get(int(bat_ids[i]), lg)
            P[i] = model.pit_rates.get(int(pit_ids[i]), lg)

    if side == "batter":
        V = B.copy()
    elif side == "pitcher":
        V = P.copy()
    elif side == "league":
        V = np.tile(lg, (n, 1))
    else:
        # Denominator must match the platoon treatment of the numerator: with
        # both sides platoon-adjusted, dividing by the hand-agnostic league
        # vector double-counts the platoon effect.
        D = np.stack([model.denominator(stands[i], hands[i], use_platoon)
                      for i in range(n)])
        V = B * P / np.maximum(D, 1e-12)

    if use_tto:
        tto = block["n_thruorder_pitcher"].fillna(1).clip(1, 4).to_numpy().astype(int)
        mult = np.stack([om.TTO_MULT.get(t, np.ones(N_OUT)) for t in tto])
        V = V * mult

    if park_factors:
        parks = block["park"].to_numpy()
        pm = np.stack([park_factors.get(p, np.ones(N_OUT)) for p in parks])
        V = V * pm

    if calibration is not None:
        V = V * calibration

    return V / V.sum(axis=1, keepdims=True)


def run(holdout, mode):
    pa = om.load_pa()
    pa = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()

    seasons = sorted(pa["season"].unique())
    if holdout not in seasons:
        raise SystemExit(f"holdout {holdout} not in data; have {seasons}")
    if not any(s < holdout for s in seasons):
        raise SystemExit(f"no prior seasons before {holdout}")

    tables = om.build_count_tables(pa)
    test = pa[pa["season"] == holdout].copy()
    test["y"] = test["outcome"].map(om.OUT_IDX)
    y = test["y"].to_numpy()

    print(f"holdout {holdout}: {len(test):,} plate appearances")
    print(f"training seasons: {[s for s in seasons if s < holdout]}")

    om.TTO_MULT = om.fit_tto(pa[pa["season"] < holdout])
    print("\ntimes-through-order multipliers (fit on prior seasons):")
    for t in [2, 3, 4]:
        pretty = "  ".join(f"{o} {om.TTO_MULT[t][i]:.3f}" for i, o in enumerate(OUTCOMES))
        print(f"  TTO {t}: {pretty}")

    park_factors = om.fit_park_factors(pa, holdout)

    # The normalization-bias correction is fit on the season before the
    # holdout, using only seasons prior to that -- so nothing from the holdout
    # reaches it. Park factors for the calibration fit are themselves fit
    # as-of that earlier season.
    calib_season = holdout - 1
    calib = None
    if calib_season in seasons:
        calib = om.fit_calibration(
            tables, pa, calib_season,
            park_factors=om.fit_park_factors(pa, calib_season))
        print(f"\nnormalization-bias correction (fit on {calib_season}):")
        print("  " + "  ".join(f"{o} {calib[i]:.3f}"
                               for i, o in enumerate(OUTCOMES)))

    if mode == "prior":
        model = om.OutcomeModel(tables, holdout, season_to_date=None)
        blocks = [(model, test)]
    else:
        blocks = []
        test["month"] = pd.to_datetime(test["game_date"]).dt.month
        std_source = pa[pa["season"] == holdout].copy()
        for month in sorted(test["month"].unique()):
            done = std_source[pd.to_datetime(std_source["game_date"]).dt.month < month]
            std = om.build_count_tables(done) if len(done) else None
            m = om.OutcomeModel(tables, holdout, season_to_date=std)
            blocks.append((m, test[test["month"] == month]))
            print(f"  built as-of model for month {month} "
                  f"({len(test[test['month'] == month]):,} PA)")

    configs = [
        ("league constant",            dict(side="league", use_platoon=False)),
        ("batter only",                dict(side="batter", use_platoon=False)),
        ("pitcher only",               dict(side="pitcher", use_platoon=False)),
        ("batter x pitcher",           dict(use_platoon=False)),
        ("+ platoon",                  dict(use_platoon=True)),
        ("+ platoon + TTO",            dict(use_platoon=True, use_tto=True)),
        ("+ platoon + TTO + park",     dict(use_platoon=True, use_tto=True,
                                            park_factors=park_factors)),
        ("+ bias correction",          dict(use_platoon=True, use_tto=True,
                                            park_factors=park_factors,
                                            calibration=calib)),
    ]

    print(f"\n{'configuration':<26} {'log loss':>10} {'Brier':>10}")
    results = {}
    for name, kw in configs:
        parts = [predict_block(m, b, **kw) for m, b in blocks]
        probs = np.vstack(parts)
        idx = np.concatenate([b["y"].to_numpy() for _, b in blocks])
        ll, bs = log_loss(probs, idx), brier(probs, idx)
        results[name] = (ll, bs, probs, idx)
        print(f"{name:<26} {ll:>10.5f} {bs:>10.5f}")

    # Oracle: the same model handed the holdout season's own data. It leaks by
    # construction and is not a legitimate score -- it exists to bound what is
    # knowable. If the honest model sits close to the oracle, the remaining
    # error is irreducible plate-appearance noise and further projection work
    # is wasted effort; if the gap is wide, the projection is leaving signal on
    # the table and is worth tuning.
    oracle_model = om.OutcomeModel(tables, holdout + 1, season_to_date=None)
    oracle_probs = predict_block(oracle_model, test, use_platoon=True,
                                 use_tto=True, park_factors=park_factors,
                                 calibration=calib)
    oracle_ll = log_loss(oracle_probs, y)
    print(f"{'oracle (leaks holdout)':<26} {oracle_ll:>10.5f} "
          f"{brier(oracle_probs, y):>10.5f}")

    base_ll = results["league constant"][0]
    full_ll = results["+ bias correction"][0]
    span = base_ll - oracle_ll
    if span > 0:
        print(f"\nshare of knowable signal captured: "
              f"{100 * (base_ll - full_ll) / span:.1f}%  "
              f"(league {base_ll:.5f} -> model {full_ll:.5f} -> oracle {oracle_ll:.5f})")
    print(f"\nimprovement over league constant: {base_ll - full_ll:.5f} nats "
          f"({100 * (base_ll - full_ll) / base_ll:.2f}%)")

    # Calibration: predicted mean rate vs observed, per outcome.
    probs, idx = results["+ bias correction"][2], results["+ bias correction"][3]
    print("\ncalibration (full model):")
    print(f"  {'outcome':<8} {'predicted':>10} {'actual':>10} {'ratio':>8}")
    for i, o in enumerate(OUTCOMES):
        pred = probs[:, i].mean()
        act = (idx == i).mean()
        print(f"  {o:<8} {pred * 100:>9.3f}% {act * 100:>9.3f}% "
              f"{pred / max(act, 1e-9):>8.4f}")

    # Reliability of the largest bucket, where miscalibration would matter most.
    print("\nreliability, strikeout probability deciles:")
    k = OUTCOMES.index("K")
    dec = pd.qcut(probs[:, k], 10, labels=False, duplicates="drop")
    for d in range(int(dec.max()) + 1):
        m = dec == d
        print(f"  decile {d}: predicted {probs[m, k].mean() * 100:5.2f}%  "
              f"actual {(idx[m] == k).mean() * 100:5.2f}%  (n={m.sum():,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=2025)
    ap.add_argument("--mode", choices=["prior", "asof"], default="prior")
    args = ap.parse_args()
    run(args.holdout, args.mode)


if __name__ == "__main__":
    main()
