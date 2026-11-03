"""Does the simulator know anything the closing line does not?

Comparing our win probabilities to the market's answers a weak question -- is
the model better than Vegas? -- and the answer is obviously no. Standalone
accuracy scores the *union* of what each side knows, so a small unique
contribution disappears under a large shared deficit.

The question worth asking is residual: conditional on everything the closing
line already prices, does our number move the probability in the right
direction? A model can be worse than the market on every metric and still
improve it, provided its errors are uncorrelated with the market's. That is the
only sense in which this project could have edge -- not by knowing more, but by
being wrong differently.

The precedent is discouraging. Market + Elo was *worse* than market alone on
2021 (0.66789 vs 0.66774), meaning Elo's information is a strict subset of what
the line prices. The simulator's value in our internal blend came entirely from
being independent of Elo; whether it is also independent of the market is what
this measures.

Blending is done in log-odds space, which is the right geometry for combining
probabilities: it is symmetric, it cannot leave [0, 1], and a fixed weight has
the same meaning at p = 0.5 as at p = 0.9.

    python market_blend.py --suffix _ship

**On sample size.** This reports a paired bootstrap because the paired
difference is the only quantity with any hope of resolution here -- both
predictions see the same games, so the shared noise cancels. Even so, at the
sample sizes available in-season the confidence interval will comfortably span
zero. Read the sign, not the significance.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def blend(p_market, p_model, w):
    """Weight `w` on the model, in log-odds space."""
    return sigmoid((1 - w) * logit(p_market) + w * logit(p_model))


def paired_bootstrap(p_a, p_b, y, n_boot=10000, seed=7):
    """Distribution of (log loss A - log loss B) resampling games together.

    Resampling the *games* rather than the predictions preserves the pairing,
    which is the whole point: the two models share almost all of their error,
    and only the difference is being asked about.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    la = -(y * np.log(np.clip(p_a, EPS, 1 - EPS)) +
           (1 - y) * np.log(np.clip(1 - p_a, EPS, 1 - EPS)))
    lb = -(y * np.log(np.clip(p_b, EPS, 1 - EPS)) +
           (1 - y) * np.log(np.clip(1 - p_b, EPS, 1 - EPS)))
    d = la - lb
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = d[idx].mean(axis=1)
    # d = (loss of A) - (loss of B), so d > 0 means B is the better model.
    return float(d.mean()), np.percentile(draws, [2.5, 97.5]), float((draws > 0).mean())


def load_frame(season, suffix, use_elo=True):
    """Games with a recorded closing line, a simulator prediction and a result."""
    cl = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "odds_closing.parquet"))
    cl = cl[["game_pk", "game_date", "home_team", "away_team",
             "p_home_close", "vig", "lead_minutes"]]

    sp = os.path.join(config.PROCESSED_DIR, f"sim_preds_{season}{suffix}.csv")
    if not os.path.exists(sp):
        raise SystemExit(f"no simulator predictions at {sp}")
    sim = pd.read_csv(sp)[["game_pk", "p_home", "home_won",
                           "sim_home_runs", "sim_away_runs",
                           "act_home_runs", "act_away_runs"]]
    sim = sim.rename(columns={"p_home": "p_sim"})

    cl["game_date"] = pd.to_datetime(cl["game_date"])
    df = cl.merge(sim, on="game_pk", how="inner")

    if use_elo:
        try:
            import baseline_models as bm
            import outcome_model as om
            pa = om.load_pa()
            elo = bm.elo_predictions(pa, season)
            if isinstance(elo, pd.DataFrame) and "p_home" in elo.columns:
                df = df.merge(elo[["game_pk", "p_home"]].rename(
                    columns={"p_home": "p_elo"}), on="game_pk", how="left")
        except Exception as exc:
            print(f"  (Elo unavailable: {exc})")

    return df.dropna(subset=["p_home_close", "p_sim", "home_won"])


def sweep(df, model_col, label, y):
    """Best blend weight, and whether it beats the market at all."""
    p_mkt = df["p_home_close"].to_numpy()
    p_mod = df[model_col].to_numpy()
    base = log_loss(p_mkt, y)

    rows = []
    for w in np.arange(0.0, 0.55, 0.05):
        p = blend(p_mkt, p_mod, w)
        rows.append((w, log_loss(p, y), brier(p, y)))

    print(f"\n  market + {label}, weight on {label}:")
    for w, ll, br in rows:
        mark = "  <- market alone" if w == 0 else ""
        print(f"    w={w:.2f}   log loss {ll:.5f}  ({ll - base:+.5f})"
              f"   Brier {br:.5f}{mark}")

    best_w, best_ll, _ = min(rows[1:], key=lambda r: r[1])
    p_best = blend(p_mkt, p_mod, best_w)
    mean_d, ci, p_better = paired_bootstrap(p_mkt, p_best, y)
    print(f"\n    best weight {best_w:.2f}: {best_ll - base:+.5f} nats vs market alone")
    print(f"    paired bootstrap of (market - blend): {mean_d:+.5f} "
          f"[{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(f"    P(blend beats market) = {p_better:.2f}")
    if best_ll >= base:
        print("    -> no weight improves on the market; nothing residual detected")
    return best_w, best_ll - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--suffix", default="_ship")
    args = ap.parse_args()

    df = load_frame(args.season, args.suffix)
    y = df["home_won"].to_numpy(dtype=float)
    n = len(df)

    print(f"=== market complementarity, {args.season} ===")
    print(f"  {n} games with a closing line, a simulator prediction and a result")
    print(f"  {df['game_date'].min().date()} to {df['game_date'].max().date()}, "
          f"median capture {df['lead_minutes'].median():.0f} min before first pitch")
    print(f"  home team won {y.mean() * 100:.1f}%")

    print("\n  standalone:")
    for col, label in [("p_home_close", "market"), ("p_sim", "simulator"),
                       ("p_elo", "Elo")]:
        if col not in df.columns or df[col].isna().all():
            continue
        p = df[col].to_numpy()
        acc = ((p > 0.5) == (y > 0.5)).mean()
        print(f"    {label:<10} log loss {log_loss(p, y):.5f}   "
              f"Brier {brier(p, y):.5f}   accuracy {acc * 100:.1f}%")

    # How much each model *disagrees* with the market is what a blend can act
    # on. Correlating raw residuals (y - p) is useless here: every prediction
    # sits near 0.5, so y - p is dominated by the shared y and the correlation
    # is forced toward 1 whether or not the models are independent. Comparing
    # the predictions themselves, in log-odds space, is the real diagnostic.
    z_mkt = logit(df["p_home_close"].to_numpy())
    print("\n  agreement with the market (log-odds):")
    print(f"    {'model':<10}{'corr':>8}{'sd(model)':>12}{'sd(market)':>12}")
    for col, label in [("p_sim", "simulator"), ("p_elo", "Elo")]:
        if col not in df.columns or df[col].isna().all():
            continue
        z = logit(df[col].fillna(0.5).to_numpy())
        print(f"    {label:<10}{np.corrcoef(z_mkt, z)[0, 1]:>+8.3f}"
              f"{z.std():>12.3f}{z_mkt.std():>12.3f}")

    # Control: shrinking the market toward 0.5 is what any under-confident
    # model does to it when blended. If plain shrinkage buys the same
    # improvement, the model contributed no information -- only timidity, which
    # pays in any sample where favourites happened to underperform.
    print("\n  control -- shrink the market toward 0.5 (no model at all):")
    base_ll = log_loss(df["p_home_close"].to_numpy(), y)
    for s in (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
        p = sigmoid(s * z_mkt)
        print(f"    shrink x{s:.2f}   log loss {log_loss(p, y):.5f}  "
              f"({log_loss(p, y) - base_ll:+.5f})")

    sweep(df, "p_sim", "simulator", y)
    if "p_elo" in df.columns and not df["p_elo"].isna().all():
        d2 = df.dropna(subset=["p_elo"])
        sweep(d2, "p_elo", "Elo", d2["home_won"].to_numpy(dtype=float))

        # Both at once: average the two models in log-odds space first, so the
        # swept weight means the same thing as in the single-model sweeps.
        df["p_both"] = sigmoid(0.5 * logit(df["p_sim"]) +
                               0.5 * logit(df["p_elo"].fillna(df["p_sim"])))
        sweep(df, "p_both", "sim+Elo", y)

    print(f"\n  NOTE: with n={n}, the resolution of this test is roughly "
          f"{1.0 / np.sqrt(n):.2f} standard errors")
    print("  against effects of order 0.005 nats. Read the sign, not the p-value.")


if __name__ == "__main__":
    main()
