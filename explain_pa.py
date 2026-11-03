"""Walk one plate appearance through every stage of the model.

Prints the numbers at each step so the chain from raw player history to a
sampled outcome is visible rather than implied.

    python explain_pa.py
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import config
import outcome_model as om
import player_ratings as pr

OUT = config.OUTCOMES


def show(label, vec, extra=""):
    print(f"  {label:<34}" + "  ".join(f"{v * 100:6.2f}" for v in vec) +
          (f"   {extra}" if extra else ""))


def main():
    pa = om.load_pa()
    season = 2026
    prior = pa[pa["season"] < season]
    tables = om.build_count_tables(pa)
    model = om.OutcomeModel(tables, season)
    lw = baserunning.linear_weights(prior, baserunning.run_expectancy(prior))

    # A high-strikeout right-hander against a good left-handed bat.
    d = pa[pa["is_pa"] & (pa["season"] == season)]
    k_rate = (d[d["is_starter"]].groupby("pitcher")["outcome"]
              .agg([("k", lambda s: (s == "K").mean()), ("n", "size")]))
    k_rate = k_rate[k_rate["n"] >= 300].sort_values("k", ascending=False)
    pitcher = int(k_rate.index[0])

    lhb = d[d["stand"] == "L"].groupby("batter").size()
    lhb = lhb[lhb >= 300].index
    vals = {int(b): float(np.dot(model.batter_vector(int(b), "L", "R"), lw))
            for b in lhb}
    batter = int(pd.Series(vals).idxmax())

    names = (pa[pa["is_pa"]].drop_duplicates("batter")
             .set_index("batter").index)
    print("=" * 78)
    print(f"ONE PLATE APPEARANCE: batter {batter} (bats L) vs "
          f"pitcher {pitcher} (throws R)")
    print("=" * 78)
    print(f"\n  {'':34}" + "  ".join(f"{o:>6}" for o in OUT))

    # --- 1. raw history -------------------------------------------------
    bat_raw = d[(d["batter"] == batter) & (d["p_throws"] == "R")]
    pit_raw = d[(d["pitcher"] == pitcher) & (d["stand"] == "L")]
    br = np.array([(bat_raw["outcome"] == o).mean() for o in OUT])
    prr = np.array([(pit_raw["outcome"] == o).mean() for o in OUT])

    print("\n1. RAW SEASON RATES (what actually happened)")
    show(f"batter vs RHP ({len(bat_raw)} PA)", br)
    show(f"pitcher vs LHB ({len(pit_raw)} PA)", prr)

    # --- 2. shrunk projections -----------------------------------------
    b = model.batter_vector(batter, "L", "R")
    p = model.pitcher_vector(pitcher, "R", "L")
    print("\n2. PROJECTED, after shrinking toward league by outcome")
    print("   (strikeouts regress with k=180 PA, triples with k=3600 --")
    print("    a small sample says a lot about one and nothing about the other)")
    show("batter projection", b)
    show("pitcher projection", p)

    # --- 3. the odds ratio ----------------------------------------------
    lg_cell = model.denominator("L", "R", use_platoon=True)
    lg = model.league
    print("\n3. LEAGUE BASELINES")
    show("league overall", lg)
    show("league, LHB vs RHP specifically", lg_cell)

    raw_v = b * p / np.maximum(lg_cell, 1e-12)
    v = raw_v / raw_v.sum()
    print("\n4. COMBINED: (batter x pitcher) / league, then normalised")
    print("   Dividing by the league cell is what stops the platoon effect")
    print("   being counted twice -- both sides already carry it.")
    show("matchup vector", v)
    print(f"\n   sanity: a league-average pitcher would return the batter's")
    print(f"   own rates, and vice versa -- that is the property the odds")
    print(f"   ratio has and simple averaging does not.")

    # --- 5. context multipliers -----------------------------------------
    om.TTO_MULT = om.fit_tto(prior)
    parks = om.fit_park_factors(pa, season)
    hfa = om.fit_home_advantage(pa, season)
    calib = om.fit_calibration(tables, pa, season - 1,
                               park_factors=om.fit_park_factors(pa, season - 1))

    print("\n5. CONTEXT MULTIPLIERS (each is a per-outcome scaling)")
    print(f"   {'':32}" + "  ".join(f"{o:>6}" for o in OUT))
    for label, mult in (("Coors Field", parks.get("COL")),
                        ("home advantage", hfa["home"]),
                        ("3rd time through the order", om.TTO_MULT[3]),
                        ("normalisation correction", calib)):
        if mult is not None:
            print(f"  {label:<32}" + "  ".join(f"{m:6.3f}" for m in mult))

    v2 = v * parks.get("COL", 1.0) * hfa["home"] * om.TTO_MULT[3] * calib
    v2 = v2 / v2.sum()
    print("\n6. AFTER CONTEXT (this batter, in Coors, at home, 3rd time through)")
    show("final outcome vector", v2)
    print(f"\n   strikeout rate moved {(v2[0] - v[0]) * 100:+.2f} points, "
          f"home run rate {(v2[7] - v[7]) * 100:+.2f}")

    # --- 6. defence ------------------------------------------------------
    oaa, speed = pr.load_ratings()
    defense = pr.DefenseRatings(oaa, pr.defensive_exposure(pa), season)
    rates = pd.Series(defense.rate)
    good = float(rates.quantile(0.95)) * 8      # a strong eight-man defence
    h = pr.defense_hit_suppression(good, model.league, lw)
    v3 = pr.apply_defense(v2, h)
    print(f"\n7. TEAM DEFENCE (a 95th-percentile glove at every position)")
    print(f"   converts {h * 100:.1f}% of balls-in-play hits into outs")
    show("after defence", v3)

    # --- 7. sampling and transition -------------------------------------
    print("\n8. SAMPLING")
    cum = np.cumsum(v3)
    print("   cumulative: " + "  ".join(f"{c:.4f}" for c in cum))
    for u in (0.15, 0.55, 0.97):
        idx = int((u >= cum).sum())
        print(f"   random draw {u:.2f}  ->  {OUT[idx]}")

    # --- 8. what the outcome does ---------------------------------------
    speed_r = pr.SpeedRatings(speed, season)
    table = baserunning.build(prior, min_count=25, speed_ratings=speed_r)
    print("\n9. WHAT THE OUTCOME DOES (runner on 2nd, 1 out, single)")
    oi = OUT.index("1B")
    for lbl, bucket in (("slow runner", 0), ("fast runner", 2)):
        c = baserunning.cell_index(oi, 2, 1, bucket, 1)
        s, n = table.starts[c], table.counts[c]
        probs = np.diff(np.concatenate([[float(c)], table.cum[s:s + n]]))
        er = float(np.dot(probs, table.runs[s:s + n]))
        scored = float(probs[table.runs[s:s + n] > 0].sum())
        print(f"   {lbl:<14} scores {scored * 100:5.1f}% of the time, "
              f"{er:.3f} runs on average")
    print("\n   These come from counting a million real plate appearances, not")
    print("   from advancement rules -- so sacrifice flies, double plays and")
    print("   runners taking an extra base are all already in the proportions")
    print("   they actually occur in.")

    print("\n" + "=" * 78)
    print("Then repeat for every plate appearance, track the game state, and")
    print("run the whole game 20,000 times. The spread of those 20,000 games")
    print("is the answer -- not any single one of them.")
    print("=" * 78)


if __name__ == "__main__":
    main()
