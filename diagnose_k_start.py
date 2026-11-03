"""Strikeout bias in the unit a prop is actually posted in: one start.

Two separate errors decide whether a projected strikeout total is right, and
they can point in opposite directions:

  rate         -- given the batters a starter faced, does the model get his
                  strikeout probability right?
  opportunity  -- does the simulator have him face the right number of batters?

`diagnose_k.py` measured the first and found strikeouts over-predicted, which
should push projections toward the over. If the model is in fact picking unders,
the second must be larger and negative -- a hook that pulls starters early costs
about 0.22 K per batter of shortfall, which swamps a rate bias of a few percent.

This separates them. The rate side reuses the PA model over the batters actually
faced; the opportunity side compares the simulator's projected batters faced
against what happened.

Also reports P(K) buckets in strikeouts per start rather than per-PA
percentages, because "the top bucket is over-predicted 10%" says nothing about
whether to bet a line posted at 6.5.

    python diagnose_k_start.py --season 2025
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om

K_IDX = config.OUTCOMES.index("K")


def monthly_models(pa, tables, season):
    """Exactly how PropSimulator builds its models: refreshed each month with
    season-to-date counts. Fitting a correction on any other configuration
    produces a constant that does not match the pipeline it is meant to fix."""
    models = {}
    for month in range(3, 11):
        done = pa[(pa["season"] == season)
                  & (pd.to_datetime(pa["game_date"]).dt.month < month)]
        std = om.build_count_tables(done) if len(done) else None
        models[month] = om.OutcomeModel(tables, season, season_to_date=std)
    return models


def pa_predictions(pa, tables, season, season_to_date=True):
    d = pa[pa["is_pa"] & (pa["season"] == season)].copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.month.clip(3, 10)

    if season_to_date:
        models = monthly_models(pa, tables, season)
    else:
        m = om.OutcomeModel(tables, season)
        models = {mo: m for mo in range(3, 11)}

    pk = np.empty(len(d))
    caches = {mo: {} for mo in models}
    for i, (b, p, st, ph, tto, mo) in enumerate(zip(
            d["batter"].to_numpy(), d["pitcher"].to_numpy(),
            d["stand"].to_numpy(), d["p_throws"].to_numpy(),
            d["n_thruorder_pitcher"].fillna(1).to_numpy(),
            d["month"].to_numpy())):
        key = (b, p, st, ph, min(int(tto), 4))
        c = caches[int(mo)]
        v = c.get(key)
        if v is None:
            v = float(models[int(mo)].matchup(b, p, st, ph, tto=int(tto))[K_IDX])
            c[key] = v
        pk[i] = v

    d["p_k"] = pk
    d["is_k"] = (d["outcome"] == "K").astype(float)
    return d


def by_start(d):
    """One row per starting pitcher appearance."""
    s = d[d["is_starter"]]
    g = (s.groupby(["game_pk", "pitcher"], as_index=False)
         .agg(proj_k=("p_k", "sum"), act_k=("is_k", "sum"),
              bf=("is_k", "size")))
    return g[g["bf"] >= 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    pa = om.load_pa()
    tables = om.build_count_tables(pa)

    for std in (False, True):
        label = "season-to-date (matches props)" if std else "prior seasons only"
        d = pa_predictions(pa, tables, args.season, season_to_date=std)
        g = by_start(d)
        print(f"\n=== {args.season}, {label} ===")
        print(f"  {len(g):,} starts, {len(d):,} PAs")
        print(f"  PA strikeout rate: predicted {d['p_k'].mean() * 100:5.2f}%"
              f"   actual {d['is_k'].mean() * 100:5.2f}%"
              f"   ({(d['p_k'].mean() - d['is_k'].mean()) / d['is_k'].mean() * 100:+.1f}%)")
        print(f"  per start, over the batters actually faced:")
        print(f"    projected K {g['proj_k'].mean():.3f}   "
              f"actual K {g['act_k'].mean():.3f}   "
              f"({g['proj_k'].mean() - g['act_k'].mean():+.3f})")
        print(f"    batters faced {g['bf'].mean():.2f}")

        if std:
            print("\n  what a P(K) bucket is worth in strikeouts per start:")
            dd = d[d["is_starter"]].copy()
            dd["bucket"] = pd.cut(dd["p_k"], [0, .15, .20, .25, .30, 1.0])
            for bk, r in dd.groupby("bucket").agg(
                    pred=("p_k", "mean"), act=("is_k", "mean"),
                    n=("is_k", "size")).iterrows():
                if r["n"] < 500:
                    continue
                bf = g["bf"].mean()
                print(f"    P(K) {str(bk):<13} {r['pred'] * bf:5.2f} projected K"
                      f"   {r['act'] * bf:5.2f} actual"
                      f"   ({(r['pred'] - r['act']) * bf:+.2f} K over "
                      f"{bf:.0f} batters)")

            print("\n  distribution of the per-start error (projected - actual):")
            e = g["proj_k"] - g["act_k"]
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                print(f"    p{int(q * 100):<3} {e.quantile(q):+.2f} K")
            print(f"    share of starts projected too high: "
                  f"{(e > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
