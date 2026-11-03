"""The running game: attempts, success, and who controls it.

Steals are already implicit in the base-out transition tables -- a plate
appearance's exit state is the next batter's entry state, and steals happen
during plate appearances, so they are bundled into every transition. The tables
are also conditioned on runner speed. So the runner side is already
represented, at least on average.

What is entirely missing is the **battery**. Nothing in the model knows that a
slow-delivery pitcher with a weak-throwing catcher is a different proposition
from a quick one with a cannon. This module measures whether that gap is worth
closing before any of it is wired into the simulator.

Two regimes: 2023 brought bigger bases and pickoff limits, which moved both
attempt and success rates sharply. Anything fitted here trains on 2023+.

    python steals.py
"""

import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

STEAL_EVENTS = {
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
}
SUCCESS_EVENTS = {"stolen_base_2b", "stolen_base_3b", "stolen_base_home"}

STEAL_PATH = os.path.join(config.PROCESSED_DIR, "steal_events.parquet")


def extract_events(seasons=None):
    """Scan the raw pitch cache for steal attempts.

    This cannot come from `pa_events`: that table keeps one row per plate
    appearance, built from the last pitch, so an attempt on pitch three of a
    six-pitch at-bat is simply absent. Steals live at pitch resolution.
    """
    paths = sorted(glob.glob(os.path.join(config.RAW_DIR, "statcast_*.parquet")))
    if seasons:
        paths = [p for p in paths
                 if int(os.path.basename(p).split("_")[1]) in seasons]

    keep = ["game_pk", "game_date", "at_bat_number", "pitch_number", "events",
            "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
            "pitcher", "fielder_2", "batter", "stand", "p_throws",
            "home_team", "away_team", "bat_score", "fld_score"]

    frames = []
    for p in paths:
        df = pd.read_parquet(p, columns=[c for c in keep if c])
        df = df[df["events"].isin(STEAL_EVENTS)]
        if len(df):
            frames.append(df)

    if not frames:
        raise SystemExit("no steal events found in the raw cache")

    ev = pd.concat(frames, ignore_index=True)
    ev["game_date"] = pd.to_datetime(ev["game_date"])
    ev["season"] = ev["game_date"].dt.year
    ev["success"] = ev["events"].isin(SUCCESS_EVENTS)

    # The runner attempting: for a steal of second it is the man on first, and
    # so on. Statcast records the pre-pitch occupancy, so the runner is read
    # off the base he left.
    base_from = np.where(ev["events"].str.contains("2b"), 1,
                         np.where(ev["events"].str.contains("3b"), 2, 3))
    ev["target_base"] = base_from + 1
    ev["runner"] = np.where(base_from == 1, ev["on_1b"],
                            np.where(base_from == 2, ev["on_2b"], ev["on_3b"]))
    ev = ev[ev["runner"].notna()].copy()
    ev["runner"] = ev["runner"].astype(int)
    ev["catcher"] = ev["fielder_2"]
    return ev


def opportunities(pa, seasons=None):
    """Plate appearances where a steal of second was available.

    Runner on first, second base open. This is the denominator the attempt rate
    needs, and it is the state the simulator will be in when it has to decide.
    """
    d = pa[pa["is_pa"]].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    opp = d[(d["base_code_before"] == 1) | (d["base_code_before"] == 5)]
    return opp


def main():
    import outcome_model as om

    print("extracting steal attempts from the raw pitch cache...")
    ev = extract_events()
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    ev.to_parquet(STEAL_PATH, index=False)
    print(f"  {len(ev):,} attempts -> {STEAL_PATH}")

    print("\nby season (the 2023 rule change is the regime break):")
    s = ev.groupby("season").agg(attempts=("success", "size"),
                                 success_rate=("success", "mean"))
    pa = om.load_pa()
    opp_by_season = (opportunities(pa).groupby("season").size()
                     .rename("opportunities"))
    s = s.join(opp_by_season)
    s["attempts_per_opp"] = s["attempts"] / s["opportunities"]
    print(s.round(4).to_string())

    recent = ev[ev["season"] >= 2023]
    opp = opportunities(pa, seasons=[2023, 2024, 2025, 2026])
    print(f"\n2023+ pooled: {len(recent):,} attempts, "
          f"success {recent['success'].mean()*100:.1f}%")

    # --- spread across runners
    r = (recent[recent["target_base"] == 2]
         .groupby("runner").agg(att=("success", "size"),
                                sr=("success", "mean")))
    r_opp = opp.groupby("r1_before").size().rename("opp")
    r = r.join(r_opp, how="left")
    r["rate"] = r["att"] / r["opp"]
    busy = r[r["opp"] >= 100]
    print(f"\nRUNNERS with 100+ opportunities: {len(busy):,}")
    print(f"  attempt rate per opportunity: mean {busy['rate'].mean():.4f}  "
          f"sd {busy['rate'].std():.4f}  "
          f"p10 {busy['rate'].quantile(.1):.4f}  p90 {busy['rate'].quantile(.9):.4f}")
    print(f"  success rate: mean {busy['sr'].mean():.3f}  sd {busy['sr'].std():.3f}")

    # --- spread across pitchers
    p = (recent[recent["target_base"] == 2]
         .groupby("pitcher").agg(att=("success", "size"),
                                 sr=("success", "mean")))
    p_opp = opp.groupby("pitcher").size().rename("opp")
    p = p.join(p_opp, how="left")
    p["rate"] = p["att"] / p["opp"]
    pbusy = p[p["opp"] >= 100]
    print(f"\nPITCHERS with 100+ opportunities: {len(pbusy):,}")
    print(f"  attempts allowed per opportunity: mean {pbusy['rate'].mean():.4f}  "
          f"sd {pbusy['rate'].std():.4f}  "
          f"p10 {pbusy['rate'].quantile(.1):.4f}  p90 {pbusy['rate'].quantile(.9):.4f}")

    # --- spread across catchers
    c = (recent[recent["target_base"] == 2]
         .groupby("catcher").agg(att=("success", "size"),
                                 sr=("success", "mean")))
    cbusy = c[c["att"] >= 50]
    print(f"\nCATCHERS with 50+ attempts against: {len(cbusy):,}")
    print(f"  success rate allowed: mean {cbusy['sr'].mean():.3f}  "
          f"sd {cbusy['sr'].std():.3f}  "
          f"p10 {cbusy['sr'].quantile(.1):.3f}  p90 {cbusy['sr'].quantile(.9):.3f}")

    # --- what is it worth?
    # Run values from the project's own expectancy matrix.
    import baserunning
    re = baserunning.run_expectancy(pa[pa["season"] >= 2023])
    # Runner on first, one out: successful steal moves 1 -> 2, failure clears
    # the base and adds an out.
    sb_val = re[2, 1] - re[1, 1]
    cs_val = re[0, 2] - re[1, 1]
    print(f"\nrun value at (runner on first, 1 out):")
    print(f"  successful steal {sb_val:+.3f}   caught stealing {cs_val:+.3f}")
    be = -cs_val / (sb_val - cs_val)
    print(f"  break-even success rate: {be*100:.1f}%")

    league_rate = len(recent[recent["target_base"] == 2]) / len(opp)
    print(f"\nleague attempt rate {league_rate:.4f} per opportunity")
    print(f"opportunities per team-game: "
          f"{len(opp) / pa.loc[pa['season'] >= 2023, 'game_pk'].nunique() / 2:.2f}")

    spread = (pbusy["rate"].quantile(.9) - pbusy["rate"].quantile(.1))
    opp_per_game = len(opp) / pa.loc[pa["season"] >= 2023, "game_pk"].nunique()
    extra_att = spread * opp_per_game
    print(f"\nPITCHER control, p90 vs p10: {spread:.4f} attempts per opportunity")
    print(f"  = {extra_att:.3f} extra attempts per game")
    print(f"  at {recent['success'].mean():.3f} success, worth "
          f"{extra_att * (recent['success'].mean() * sb_val + (1 - recent['success'].mean()) * cs_val):+.4f} "
          f"runs per game")


if __name__ == "__main__":
    main()
