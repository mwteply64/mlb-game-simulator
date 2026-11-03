"""Cross-validated search over the outcome model's constants.

Scored on held-out plate appearance log loss, leave-one-season-out, with a
candidate required to win in a MAJORITY of folds rather than on the mean. The
mean alone is dominated by whichever season happens to be easiest; the fold
count is what separates a real effect from one good year.

Two guards against the failure mode that killed most of the lineup project's
post-v13 candidates:

* **Staged search.** A single shared multiplier on all eight batter regression
  constants is fitted before any of them is touched individually. One parameter
  cannot overfit the way eight can, and if the shared scale explains the gain
  then the individual sweep has nothing left to find.
* **Nothing ships on plate appearance log loss alone.** Every winner is
  re-verified end to end in the simulator. This project has repeatedly produced
  component gains that did not reach the game level -- defence and baserunning
  both improved their own metrics and moved simulator AUC not at all.

    python tuner_outcome.py --stage scale
    python tuner_outcome.py --stage individual
"""

import argparse
import copy
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import outcome_model as om

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)
FOLDS = [2023, 2024, 2025]
RESULTS = os.path.join(config.PROJECT_DIR, "tuner_outcome_results.json")


def baseline_params():
    return {
        "REGRESS_PA_BATTER": dict(om.REGRESS_PA_BATTER),
        "REGRESS_PA_PITCHER": dict(om.REGRESS_PA_PITCHER),
        "DECAY_PER_SEASON_BATTER": om.DECAY_PER_SEASON_BATTER,
        "DECAY_PER_SEASON_PITCHER": om.DECAY_PER_SEASON_PITCHER,
        "PLATOON_PRIOR_PA_BATTER": om.PLATOON_PRIOR_PA_BATTER,
        "PLATOON_PRIOR_PA_PITCHER": om.PLATOON_PRIOR_PA_PITCHER,
        "LEAGUE_NOWCAST_PA": om.LEAGUE_NOWCAST_PA,
        "MIN_PA_FOR_OWN_RATES": om.MIN_PA_FOR_OWN_RATES,
    }


class FoldData:
    """Everything a fold needs, precomputed once so the search is fast.

    Player identities are mapped to integer slots up front, which turns the
    per-plate-appearance prediction into array indexing instead of a
    dictionary lookup in a Python loop -- the difference between a minute per
    candidate and a couple of seconds.
    """

    def __init__(self, pa, tables, season):
        self.season = season
        self.tables = tables
        test = pa[(pa["season"] == season) & pa["is_pa"]
                  & pa["outcome"].isin(OUTCOMES)]
        self.y = test["outcome"].map(om.OUT_IDX).to_numpy()

        self.bat = test["batter"].to_numpy().astype(int)
        self.pit = test["pitcher"].to_numpy().astype(int)
        self.stand = test["stand"].to_numpy()
        self.hand = test["p_throws"].to_numpy()

        # Distinct (player, side, hand) combinations actually needed.
        bat_keys = pd.MultiIndex.from_arrays(
            [self.bat, self.stand, self.hand]).unique()
        pit_keys = pd.MultiIndex.from_arrays(
            [self.pit, self.hand, self.stand]).unique()
        self.bat_keys = list(bat_keys)
        self.pit_keys = list(pit_keys)

        bmap = {k: i for i, k in enumerate(self.bat_keys)}
        pmap = {k: i for i, k in enumerate(self.pit_keys)}
        self.bat_slot = np.array(
            [bmap[k] for k in zip(self.bat, self.stand, self.hand)])
        self.pit_slot = np.array(
            [pmap[k] for k in zip(self.pit, self.hand, self.stand)])

        cell_keys = list(pd.MultiIndex.from_arrays(
            [self.stand, self.hand]).unique())
        cmap = {k: i for i, k in enumerate(cell_keys)}
        self.cell_keys = cell_keys
        self.cell_slot = np.array(
            [cmap[k] for k in zip(self.stand, self.hand)])

    def log_loss(self, params):
        model = om.OutcomeModel(self.tables, self.season, params=params)

        B = np.stack([model.batter_vector(int(p), s, h)
                      for p, s, h in self.bat_keys])
        P = np.stack([model.pitcher_vector(int(p), h, s)
                      for p, h, s in self.pit_keys])
        D = np.stack([model.denominator(s, h, use_platoon=True)
                      for s, h in self.cell_keys])

        V = (B[self.bat_slot] * P[self.pit_slot]
             / np.maximum(D[self.cell_slot], 1e-12))
        V /= V.sum(axis=1, keepdims=True)
        p = np.maximum(V[np.arange(len(self.y)), self.y], 1e-12)
        return float(-np.mean(np.log(p)))


def evaluate(folds, params):
    return np.array([f.log_loss(params) for f in folds])


def report(name, base, cand):
    wins = int((cand < base).sum())
    delta = float((base - cand).mean())
    per = "  ".join(f"{b - c:+.5f}" for b, c in zip(base, cand))
    flag = "SHIP" if wins >= len(base) - 0 else ("mixed" if wins else "no")
    print(f"  {name:<28} mean {delta:+.5f}  folds {wins}/{len(base)}  "
          f"[{per}]  {flag}")
    return delta, wins


def stage_scale(folds, base_params, base_scores):
    print("\n=== stage 1: shared multiplier on the regression constants ===")
    best = {"batter": 1.0, "pitcher": 1.0}
    for side, key in (("batter", "REGRESS_PA_BATTER"),
                      ("pitcher", "REGRESS_PA_PITCHER")):
        print(f"\n{side} constants:")
        results = {}
        # Bracketed rather than open-ended: the first pass ran to the edge of
        # its grid, and an optimum at the boundary is not an optimum.
        for mult in (0.7, 1.0, 1.6, 2.2, 3.0, 4.0, 6.0):
            p = copy.deepcopy(base_params)
            p[key] = {k: max(v * mult, 1.0) for k, v in base_params[key].items()}
            sc = evaluate(folds, p)
            d, w = report(f"x{mult}", base_scores, sc)
            results[mult] = (d, w)
        good = {m: v for m, v in results.items() if v[1] == len(folds)}
        if good:
            best[side] = max(good, key=lambda m: good[m][0])
        print(f"  -> best {side} multiplier winning every fold: x{best[side]}")
    return best


def stage_individual(folds, base_params, base_scores):
    print("\n=== stage 2: per-outcome coordinate ascent ===")
    params = copy.deepcopy(base_params)
    cur = base_scores.copy()

    for key in ("REGRESS_PA_BATTER", "REGRESS_PA_PITCHER"):
        for outcome in OUTCOMES:
            start = params[key][outcome]
            best_val, best_delta = start, 0.0
            for mult in (0.6, 0.8, 1.25, 1.6):
                trial = copy.deepcopy(params)
                trial[key][outcome] = max(start * mult, 1.0)
                sc = evaluate(folds, trial)
                wins = int((sc < cur).sum())
                delta = float((cur - sc).mean())
                if wins == len(folds) and delta > best_delta:
                    best_val, best_delta = trial[key][outcome], delta
            if best_val != start:
                params[key][outcome] = best_val
                cur = evaluate(folds, params)
                print(f"  {key}[{outcome}] {start:.0f} -> {best_val:.0f}  "
                      f"(mean {best_delta:+.5f}, all folds)")
            else:
                print(f"  {key}[{outcome}] {start:.0f} held")
    return params, cur


def stage_scalars(folds, params, cur):
    print("\n=== stage 3: decay, platoon priors, nowcast ===")
    grids = {
        "DECAY_PER_SEASON_BATTER": [0.70, 0.80, 0.85, 0.92, 1.0],
        "DECAY_PER_SEASON_PITCHER": [0.65, 0.75, 0.80, 0.88, 1.0],
        "PLATOON_PRIOR_PA_BATTER": [150, 250, 400, 700, 1200],
        "PLATOON_PRIOR_PA_PITCHER": [200, 350, 500, 900, 1500],
        "LEAGUE_NOWCAST_PA": [8000, 15000, 30000, 60000],
        "MIN_PA_FOR_OWN_RATES": [10, 25, 50, 100],
    }
    for key, grid in grids.items():
        start = params[key]
        best_val, best_delta = start, 0.0
        for v in grid:
            if v == start:
                continue
            trial = copy.deepcopy(params)
            trial[key] = v
            sc = evaluate(folds, trial)
            wins = int((sc < cur).sum())
            delta = float((cur - sc).mean())
            if wins == len(folds) and delta > best_delta:
                best_val, best_delta = v, delta
        if best_val != start:
            params[key] = best_val
            cur = evaluate(folds, params)
            print(f"  {key} {start} -> {best_val}  "
                  f"(mean {best_delta:+.5f}, all folds)")
        else:
            print(f"  {key} {start} held")
    return params, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["scale", "individual", "scalars", "all"])
    args = ap.parse_args()

    pa = om.load_pa()
    tables = om.build_count_tables(pa)
    print("preparing folds...")
    folds = [FoldData(pa, tables, s) for s in FOLDS]
    for f in folds:
        print(f"  {f.season}: {len(f.y):,} plate appearances")

    base = baseline_params()
    t0 = time.time()
    base_scores = evaluate(folds, base)
    print(f"\nbaseline log loss per fold: "
          f"{'  '.join(f'{s:.5f}' for s in base_scores)}  "
          f"(mean {base_scores.mean():.5f})")
    print(f"one evaluation takes {(time.time() - t0) / len(folds):.1f}s per fold")

    out = {"baseline": {"scores": base_scores.tolist(),
                        "mean": float(base_scores.mean())}}

    params, cur = copy.deepcopy(base), base_scores.copy()
    if args.stage in ("scale", "all"):
        best = stage_scale(folds, base, base_scores)
        for side, key in (("batter", "REGRESS_PA_BATTER"),
                          ("pitcher", "REGRESS_PA_PITCHER")):
            if best[side] != 1.0:
                params[key] = {k: max(v * best[side], 1.0)
                               for k, v in base[key].items()}
        cur = evaluate(folds, params)
        out["after_scale"] = {"mult": best, "scores": cur.tolist(),
                              "mean": float(cur.mean())}
        print(f"\nafter stage 1: mean {cur.mean():.5f} "
              f"(gain {base_scores.mean() - cur.mean():+.5f})")

    if args.stage in ("individual", "all"):
        params, cur = stage_individual(folds, params, cur)
        out["after_individual"] = {"scores": cur.tolist(),
                                   "mean": float(cur.mean())}

    if args.stage in ("scalars", "all"):
        params, cur = stage_scalars(folds, params, cur)
        out["after_scalars"] = {"scores": cur.tolist(),
                                "mean": float(cur.mean())}

    out["final_params"] = params
    total = base_scores.mean() - cur.mean()
    print(f"\nTOTAL: {base_scores.mean():.5f} -> {cur.mean():.5f} "
          f"({total:+.5f} nats, {100 * total / base_scores.mean():.3f}%)")
    print("Verify in the simulator before shipping -- plate appearance gains "
          "have not reliably reached the game level in this project.")

    with open(RESULTS, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
