"""Pinch hitting for the bulk simulator.

The game simulator has always used a fixed nine-slot lineup: pitchers turned
over, batters never did. Measured on 2025 that omission is worth 3.74% of all
plate appearances -- 14.1% of ninth-inning PAs, 18.6% in extras -- and 11.0% of
lineup slots finish the game with a substitute in them. For win probability that
is second order. For hitter props it is not: a starter's plate appearance count
is overstated by about 2.6%, which inflates every counting prop.

Two pieces, mirroring how the pitching hook already works:

* **When.** A fitted hazard evaluated after every plate appearance, driven by
  leverage rather than a fixed inning rule -- the same idea as `HazardHook`.
  Managers do not pinch hit on a timer; they spend a bench bat when the spot is
  worth spending it on, which is what leverage measures.

* **Who.** The league-average pinch hitter's outcome vector, split by platoon
  state, taken empirically from real substitute plate appearances the same way
  `league_reliever_vectors` handles relief pitching. Modelling bench players
  individually would need per-game bench rosters and would add far more variance
  than the effect being captured.

A substitution is permanent for the slot: once a bench bat enters, that slot
keeps his rates for the rest of the game, which is what actually happens and
what makes the bench a depleting resource.

    python pinch_hitting.py --fit
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import leverage as lev

OUTCOMES = config.OUTCOMES
PATH = os.path.join(config.PROCESSED_DIR, "pinch_hazard.json")

# Features the hazard is fitted on. Deliberately few: this fires up to nine
# times a game across two teams inside the simulation loop, so every term has to
# be computable from state already being tracked.
FEATURES = ["late", "very_late", "log_li", "platoon_disadv", "trailing_close",
            "pitcher_slot_gt0"]


def substitution_frame(pa, seasons=None):
    """Discrete-time hazard frame: is this plate appearance the substitution?

    The event is attached to the plate appearance where the substitute actually
    bats, because that is when the decision is made. An earlier version defined
    the target as "the *next* PA in this slot is a substitute", which attributed
    the decision to the starter's previous time up -- often several innings
    earlier -- and censored every substitution where the slot never came around
    again. It fitted a coefficient of -2.22 on `very_late` and put
    ninth-inning substitutions at 0.35% when the true figure is 14.1%,
    i.e. exactly backwards.

    Risk set: every plate appearance in a slot up to and including the first
    substitute. Once a slot has turned over it leaves the risk set, since a
    second substitution is a different and much rarer decision.
    """
    d = pa[pa["is_pa"]].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    d = d.sort_values(["game_pk", "at_bat_number"])

    key = ["game_pk", "batting_team", "lineup_slot"]
    first = d.drop_duplicates(subset=key, keep="first")[key + ["batter"]]
    first = first.rename(columns={"batter": "starter_id"})
    d = d.merge(first, on=key, how="left")
    d["is_sub"] = d["batter"] != d["starter_id"]

    # Keep everything up to the first substitute inclusive: cumulative count of
    # substitute PAs in the slot is 0 for the starter's turns and 1 on the
    # substitution itself.
    d["sub_seen"] = d.groupby(key)["is_sub"].cumsum()
    d = d[(d["sub_seen"] == 0) | ((d["sub_seen"] == 1) & d["is_sub"])]
    d["target"] = d["is_sub"].astype(float)

    # The incumbent's handedness, not the batter who actually appeared. On an
    # event row `stand` belongs to the substitute, and substitutes are picked
    # for the platoon advantage -- using it directly taught the model that
    # substitutions happen when the platoon matchup is already favourable, a
    # coefficient of -1.04 on exactly the term that should be positive. What
    # the manager is looking at is the starter's matchup.
    st = d.drop_duplicates(subset=key, keep="first")[key + ["stand"]]
    st = st.rename(columns={"stand": "starter_stand"})
    d = d.merge(st, on=key, how="left")
    return d.copy()


def add_features(d, wp=None):
    """Feature columns, shared by the fit and by the simulator."""
    inning = d["inning"].to_numpy()
    d["late"] = (inning >= 7).astype(float)
    d["very_late"] = (inning >= 9).astype(float)

    margin = (d["bat_score_before"] - d["fld_score_before"]).to_numpy()
    d["trailing_close"] = ((margin < 0) & (margin >= -3)).astype(float)

    # Platoon disadvantage for the *incumbent*, the classic reason to spend a
    # bench bat. Switch hitters never carry one.
    stand = (d["starter_stand"] if "starter_stand" in d.columns
             else d["stand"]).to_numpy()
    throws = d["p_throws"].to_numpy()
    d["platoon_disadv"] = ((stand == throws) & (stand != "S")).astype(float)

    # A reliever already being in is a strong signal: the manager has committed
    # to late-game management, and the platoon matchup is now fixed.
    d["pitcher_slot_gt0"] = (~d["is_starter"]).astype(float) \
        if "is_starter" in d.columns else 0.0

    # Precomputed leverage, indexed rather than recomputed. `leverage()` is
    # scalar and walks the transition table per call; indexing the cached table
    # gives the same numbers at no per-row cost.
    li_table = lev.load_leverage_table()
    if li_table is not None:
        idx = lev.state_index(inning, d["is_top"].to_numpy(),
                              d["outs_before"].to_numpy(),
                              d["base_code_before"].to_numpy(), margin)
        li = li_table[np.clip(idx, 0, len(li_table) - 1)]
    else:
        li = np.ones(len(d))
    d["log_li"] = np.log(np.clip(li, 0.05, 20.0))
    return d


def fit_logistic(X, y, iters=200, l2=1e-4):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (y - p) - l2 * np.r_[0.0, w[1:]]
        s = np.maximum(p * (1 - p), 1e-9)
        H = (X * s[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--seasons", type=int, nargs="*")
    args = ap.parse_args()

    import outcome_model as om
    pa = om.load_pa()
    seasons = args.seasons or [s for s in config.SEASONS if s <= 2025]

    d = substitution_frame(pa, seasons)
    d = add_features(d)
    y = d["target"].to_numpy(dtype=float)
    X = d[FEATURES].to_numpy(dtype=float)
    w = fit_logistic(X, y)

    print(f"fitted on {len(d):,} non-substitute PAs, "
          f"{y.mean() * 100:.2f}% followed by a substitution")
    print(f"  intercept {w[0]:+.4f}")
    for name, c in zip(FEATURES, w[1:]):
        print(f"  {name:<18} {c:+.4f}")

    p = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))
    print(f"\n  mean predicted {p.mean() * 100:.3f}%  actual {y.mean() * 100:.3f}%")
    d = d.assign(p=p)
    print("\n  calibration by predicted decile:")
    d["dec"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    g = d.groupby("dec").agg(pred=("p", "mean"), act=("target", "mean"),
                             n=("target", "size"))
    for i, r in g.iterrows():
        print(f"    d{int(i)}  predicted {r['pred'] * 100:5.2f}%  "
              f"actual {r['act'] * 100:5.2f}%  (n={int(r['n']):,})")

    print("\n  by inning (predicted vs actual):")
    gi = d.groupby("inning").agg(pred=("p", "mean"), act=("target", "mean"),
                                 n=("target", "size"))
    for i, r in gi.iterrows():
        if r["n"] < 500 or i > 10:
            continue
        print(f"    inning {int(i):2d}  predicted {r['pred'] * 100:5.2f}%  "
              f"actual {r['act'] * 100:5.2f}%")

    with open(PATH, "w") as fh:
        json.dump({"features": FEATURES, "weights": w.tolist(),
                   "seasons": seasons}, fh, indent=2)
    print(f"\nwrote {PATH}")


def load_hazard():
    if not os.path.exists(PATH):
        return None
    try:
        with open(PATH) as fh:
            blob = json.load(fh)
        return blob["features"], np.array(blob["weights"], dtype=float)
    except Exception:
        return None


def pinch_hitter_vectors(pa, seasons=None):
    """League-average pinch hitter outcome rates, by batter side.

    Taken from real substitute plate appearances rather than assuming a bench
    bat hits like a starter. Bench players are worse hitters on average, but
    they enter with the platoon advantage far more often than not, and those two
    effects partly cancel -- which is exactly why it is worth measuring instead
    of guessing.
    """
    d = substitution_frame_all(pa, seasons)
    out = {}
    for side in ["L", "R"]:
        s = d[(d["stand"] == side) & d["outcome"].isin(OUTCOMES)]
        v = np.array([(s["outcome"] == o).sum() for o in OUTCOMES], dtype=float)
        out[side] = v / v.sum() if v.sum() else None
    return out


def substitution_frame_all(pa, seasons=None):
    """Every plate appearance taken by a substitute."""
    d = pa[pa["is_pa"]].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    key = ["game_pk", "batting_team", "lineup_slot"]
    d = d.sort_values(["game_pk", "at_bat_number"])
    first = d.drop_duplicates(subset=key, keep="first")[key + ["batter"]]
    first = first.rename(columns={"batter": "starter_id"})
    d = d.merge(first, on=key, how="left")
    return d[d["batter"] != d["starter_id"]]


if __name__ == "__main__":
    main()
