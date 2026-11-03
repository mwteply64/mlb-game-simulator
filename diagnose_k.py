"""Is the strikeout rate the model predicts the strikeout rate that happens?

Found while testing the arsenal hypothesis: on 2025 the PA model predicted a
23.11% strikeout rate against an actual 22.22%. That is a 4% relative
over-prediction, and unlike a win-probability bias it does not cancel -- both
teams' strikeouts are inflated in the same direction. A starter facing 24
batters is projected for 5.55 strikeouts instead of 5.33, which on a line
posted at 5.5 is the difference between a bet and no bet, in the same direction
every time.

Strikeout props are the part of this project most likely to be worth anything,
so a standing bias in exactly that quantity is worth more than any feature.

    python diagnose_k.py
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om

K_IDX = config.OUTCOMES.index("K")


def predicted_k(pa, tables, season):
    model = om.OutcomeModel(tables, season)
    d = pa[pa["is_pa"] & (pa["season"] == season)]
    cache, pk = {}, np.empty(len(d))
    for i, (b, p, st, ph, tto) in enumerate(zip(
            d["batter"].to_numpy(), d["pitcher"].to_numpy(),
            d["stand"].to_numpy(), d["p_throws"].to_numpy(),
            d["n_thruorder_pitcher"].fillna(1).to_numpy())):
        key = (b, p, st, ph, min(int(tto), 4))
        v = cache.get(key)
        if v is None:
            v = float(model.matchup(b, p, st, ph, tto=int(tto))[K_IDX])
            cache[key] = v
        pk[i] = v
    return d.assign(p_k=pk, is_k=(d["outcome"] == "K").astype(float))


def main():
    pa = om.load_pa()
    tables = om.build_count_tables(pa)

    print("=== predicted vs actual strikeout rate, by season ===")
    print("  (model built as-of each season; strictly prior data plus "
          "season-to-date)\n")
    rows = []
    for season in [s for s in config.SEASONS if s >= 2022]:
        d = predicted_k(pa, tables, season)
        if not len(d):
            continue
        pred, act = d["p_k"].mean(), d["is_k"].mean()
        rel = (pred - act) / act * 100
        print(f"  {season}   predicted {pred * 100:5.2f}%   actual {act * 100:5.2f}%"
              f"   {rel:+5.1f}%   ({len(d):,} PAs)")
        rows.append((season, d))

        st = d[d["is_starter"]] if "is_starter" in d.columns else None
        if st is not None and len(st):
            print(f"          starters only:  predicted {st['p_k'].mean() * 100:5.2f}%"
                  f"   actual {st['is_k'].mean() * 100:5.2f}%"
                  f"   {(st['p_k'].mean() - st['is_k'].mean()) / st['is_k'].mean() * 100:+5.1f}%")

    # What the bias is worth on an actual prop.
    d = rows[-1][1]
    bias = d["p_k"].mean() - d["is_k"].mean()
    print(f"\n  on a 24-batter start, the {rows[-1][0]} bias is worth "
          f"{bias * 24:+.2f} strikeouts")
    print("  every projection, in the same direction -- this is a bet-sizing "
          "bias, not noise")

    # Where it lives: is it uniform, or concentrated in the pitchers whose
    # props you would actually bet?
    print("\n=== is the bias uniform across the prediction range? ===")
    d = d.copy()
    d["bucket"] = pd.cut(d["p_k"], [0, .15, .20, .25, .30, 1.0])
    g = d.groupby("bucket").agg(pred=("p_k", "mean"), act=("is_k", "mean"),
                                n=("is_k", "size"))
    for b, r in g.iterrows():
        if r["n"] < 500:
            continue
        print(f"  P(K) {str(b):<14} predicted {r['pred'] * 100:5.2f}%   "
              f"actual {r['act'] * 100:5.2f}%   "
              f"{(r['pred'] - r['act']) / r['act'] * 100:+5.1f}%   (n={int(r['n']):,})")


if __name__ == "__main__":
    main()
