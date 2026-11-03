"""Collapse Statcast pitch rows into one row per plate appearance.

The output table is the substrate for everything downstream: the outcome model
trains on it, the baserunning transition tables are counted from it, and the
hook / bullpen models read their features off it.

Two constructions here are easy to get subtly wrong, so they are spelled out:

1. **Entry state comes from the FIRST pitch of the at-bat, the event from the
   LAST.** Runners steal mid-plate-appearance, so the base state on the final
   pitch is not the state the batter came up in. Using the last pitch for both
   silently attributes stolen bases to the batter's outcome.

2. **Exit state is the entry state of the next at-bat in the same half-inning.**
   Baseball has no dead time between plate appearances -- every steal, wild
   pitch, and balk happens during some batter's at-bat -- so the next batter's
   entry state is exactly the state this plate appearance left behind.

Run:
    python build_pa_events.py
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------
# Event taxonomy
# --------------------------------------------------------------------------

# Events that end a plate appearance and therefore advance the batting order.
EVENT_TO_OUTCOME = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "strikeout_triple_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "field_out": "OUT",
    "force_out": "OUT",
    "grounded_into_double_play": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "sac_fly": "OUT",
    "sac_bunt": "OUT",
    "sac_fly_double_play": "OUT",
    "sac_bunt_double_play": "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "other_out": "OUT",
    "batter_interference": "OUT",
    "field_error": "OUT",       # see ROE note below
    "catcher_interf": "OTHER",  # batter reaches, no at-bat charged; rare
}

# Reached-on-error is classified OUT for the *batter* outcome model on purpose.
# Batter skill has almost no influence on whether a fielder muffs the ball, so
# folding ROE into the batter's hit rates would attribute defense to offense.
# The `is_roe` flag preserved here lets the simulator apply errors as a
# defense-driven overlay: an OUT is converted to a reached-base with a
# probability that belongs to the fielding team, not the hitter.
ROE_EVENTS = {"field_error"}

# Events that can appear on a pitch without ending the plate appearance (or
# that end an at-bat without completing it, e.g. an inning-ending caught
# stealing). These must not advance the batting order.
NON_PA_EVENTS = {
    "truncated_pa", "caught_stealing_2b", "caught_stealing_3b",
    "caught_stealing_home", "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "stolen_base_2b", "stolen_base_3b",
    "stolen_base_home", "wild_pitch", "passed_ball", "balk", "other_advance",
    "runner_double_play", "defensive_indiff", "error", "game_advisory",
    "ejection", "cs_double_play", "pickoff_error_1b", "pickoff_error_2b",
    "pickoff_error_3b", "sac_fly_double_play_error",
}


def load_raw(seasons=None):
    paths = sorted(glob.glob(os.path.join(config.RAW_DIR, "statcast_*.parquet")))
    if seasons:
        paths = [p for p in paths
                 if int(os.path.basename(p).split("_")[1]) in seasons]
    if not paths:
        raise SystemExit(f"no raw chunks in {config.RAW_DIR}; run fetch_statcast.py first")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df):,} pitches from {len(paths)} chunks")
    return df


# Columns that must be numeric. Statcast occasionally returns one of these as
# object dtype in a chunk where it is entirely null, and concatenating that
# chunk with the rest poisons the column for the whole span -- which then
# surfaces much later as a groupby failure rather than at load.
NUMERIC_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "inning",
    "outs_when_up", "balls", "strikes", "bat_score", "fld_score",
    "post_bat_score", "home_score", "away_score", "post_home_score",
    "post_away_score", "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game",
    "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value", "delta_run_exp",
    "delta_home_win_exp", "release_speed", "hit_location",
]


def build(df):
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year

    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Half-inning key. inning_topbot is TOP when the away team bats.
    df["is_top"] = df["inning_topbot"].str.upper().str[0].eq("T")
    df = df.sort_values(["game_pk", "inning", "is_top", "at_bat_number",
                         "pitch_number"], ascending=[True, True, False, True, True])

    ab_key = ["game_pk", "at_bat_number"]
    # drop_duplicates, NOT groupby().first()/.last(). GroupBy.first returns the
    # first non-null value of each column *independently*, so a runner who stole
    # third mid-at-bat would populate on_3b for the at-bat's entry state and
    # make it look like third base was occupied when the batter stepped in.
    # These must be the actual first and last pitch rows, nulls included.
    first = df.drop_duplicates(subset=ab_key, keep="first").reset_index(drop=True)
    last = df.drop_duplicates(subset=ab_key, keep="last").reset_index(drop=True)

    fielder_cols = [f"fielder_{i}" for i in range(2, 10)
                    if f"fielder_{i}" in first.columns]

    pa = first[[
        "game_pk", "game_date", "season", "at_bat_number", "inning", "is_top",
        "home_team", "away_team", "batter", "pitcher", "stand", "p_throws",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
        "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
        "pitcher_days_since_prev_game", "batter_days_since_prev_game",
        "if_fielding_alignment", "of_fielding_alignment",
    ] + fielder_cols].rename(columns={
        "outs_when_up": "outs_before",
        "on_1b": "r1_before", "on_2b": "r2_before", "on_3b": "r3_before",
        "bat_score": "bat_score_before", "fld_score": "fld_score_before",
    })

    tail = last[[
        "game_pk", "at_bat_number", "events", "description", "des",
        "post_bat_score", "bb_type", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "pitch_number", "delta_run_exp", "delta_home_win_exp",
    ]].rename(columns={"pitch_number": "pitches_in_pa"})

    pa = pa.merge(tail, on=ab_key, how="left")

    pa["raw_event"] = pa["events"].fillna("none")
    pa["outcome"] = pa["raw_event"].map(EVENT_TO_OUTCOME)
    pa["is_roe"] = pa["raw_event"].isin(ROE_EVENTS)
    pa["is_pa"] = pa["outcome"].notna()

    unknown = sorted(set(pa.loc[~pa["is_pa"], "raw_event"]) - NON_PA_EVENTS - {"none"})
    if unknown:
        print(f"WARNING: unclassified events treated as non-PA: {unknown}")

    # Teams and park.
    pa["batting_team"] = np.where(pa["is_top"], pa["away_team"], pa["home_team"])
    pa["fielding_team"] = np.where(pa["is_top"], pa["home_team"], pa["away_team"])
    pa["park"] = pa["home_team"]
    pa["batting_is_home"] = ~pa["is_top"]

    pa = pa.sort_values(["game_pk", "inning", "is_top", "at_bat_number"],
                        ascending=[True, True, False, True]).reset_index(drop=True)

    # ---- exit state: entry state of the next at-bat in the same half-inning
    pa["half_key"] = (pa["game_pk"].astype(str) + "_" + pa["inning"].astype(str)
                      + "_" + pa["is_top"].astype(str))
    grp = pa.groupby("half_key", sort=False)

    for src, dst in [("outs_before", "outs_after"), ("r1_before", "r1_after"),
                     ("r2_before", "r2_after"), ("r3_before", "r3_after")]:
        pa[dst] = grp[src].shift(-1)

    # Last plate appearance of a half-inning: three outs, bases cleared. The
    # exception is a walk-off, where the half-inning ends on the run, not an
    # out -- outs_after is then whatever the game ended with, so it is left as
    # the pre-PA outs plus any outs the play itself recorded. Walk-offs are
    # flagged rather than guessed at.
    last_of_half = pa["outs_after"].isna()
    pa["ends_half"] = last_of_half
    pa.loc[last_of_half, "outs_after"] = 3.0
    for c in ["r1_after", "r2_after", "r3_after"]:
        pa.loc[last_of_half, c] = np.nan

    # ---- runs and outs charged to this plate appearance
    # Runs are taken as the difference between this plate appearance's entry
    # score and the batting team's NEXT entry score, rather than from
    # post_bat_score. Statcast occasionally omits a run from the final pitch of
    # the play that scored it and reflects it only on the following at-bat, so
    # differencing post_bat_score loses roughly one run per few hundred games.
    # Differencing consecutive entry scores telescopes exactly by construction:
    # summing over a team-game always reproduces the final score. The cost is
    # that in those rare cases the run is attributed to the following plate
    # appearance, which is a placement error of one batter, not a lost run.
    pa["_team_seq"] = pa["game_pk"].astype(str) + "_" + pa["batting_team"]
    next_entry = pa.groupby("_team_seq")["bat_score_before"].shift(-1)
    pa["runs_pa"] = (next_entry - pa["bat_score_before"])
    last_of_team_game = next_entry.isna()
    pa.loc[last_of_team_game, "runs_pa"] = (
        pa.loc[last_of_team_game, "post_bat_score"]
        - pa.loc[last_of_team_game, "bat_score_before"])
    pa["runs_pa"] = pa["runs_pa"].fillna(0).clip(lower=0)

    pa["outs_pa"] = (pa["outs_after"] - pa["outs_before"]).clip(lower=0)

    for c in ["r1_before", "r2_before", "r3_before"]:
        pa[c.replace("_before", "_occ_before")] = pa[c].notna()
    for c in ["r1_after", "r2_after", "r3_after"]:
        pa[c.replace("_after", "_occ_after")] = pa[c].notna()

    # Base-out state code, the Markov state the simulator moves through:
    # 0-7 base occupancy (1B=1, 2B=2, 3B=4) crossed with 0-2 outs.
    pa["base_code_before"] = (pa["r1_occ_before"].astype(int)
                              + 2 * pa["r2_occ_before"].astype(int)
                              + 4 * pa["r3_occ_before"].astype(int))
    pa["base_code_after"] = np.where(
        pa["ends_half"], -1,
        pa["r1_occ_after"].astype(int) + 2 * pa["r2_occ_after"].astype(int)
        + 4 * pa["r3_occ_after"].astype(int))
    pa["state_before"] = pa["base_code_before"] * 3 + pa["outs_before"].astype(int)

    # ---- batting order slot
    # Only order-advancing plate appearances count, so a caught stealing that
    # ends an inning mid-at-bat does not shift the whole lineup by one. Slot is
    # then position in the cycle, which handles substitutions for free: a pinch
    # hitter inherits the slot of the man he replaced.
    pa["team_key"] = pa["game_pk"].astype(str) + "_" + pa["batting_team"]
    pa["pa_idx"] = pa.groupby("team_key")["is_pa"].cumsum() - 1
    pa["lineup_slot"] = np.where(pa["is_pa"], (pa["pa_idx"] % 9) + 1, np.nan)

    # ---- pitcher appearance tracking (feeds the hook / bullpen models)
    pa["pitcher_key"] = pa["game_pk"].astype(str) + "_" + pa["pitcher"].astype(str)
    pa["pitcher_bf"] = pa.groupby("pitcher_key")["is_pa"].cumsum()
    pa["pitcher_pitches"] = pa.groupby("pitcher_key")["pitches_in_pa"].cumsum()
    # The starter is the pitcher who threw his team's first at-bat. Comparing
    # against the *game's* first at-bat instead only ever finds the home
    # starter -- the away starter debuts in the bottom of the first, several
    # at-bats later -- which silently reclassifies every away starter as a
    # reliever.
    pitcher_first_ab = pa.groupby("pitcher_key")["at_bat_number"].transform("min")
    team_first_ab = pa.groupby(["game_pk", "fielding_team"])["at_bat_number"].transform("min")
    pa["is_starter"] = pitcher_first_ab == team_first_ab

    pa["platoon"] = pa["stand"].astype(str) + "v" + pa["p_throws"].astype(str)
    pa["score_diff_before"] = pa["bat_score_before"] - pa["fld_score_before"]

    return pa


def validate(pa):
    """Correctness gates. A silent failure in any of these poisons everything
    downstream, so they run every build and print rather than assert."""
    print("\n--- validation ---")
    ok = True

    # 1. Reconstructed final scores must match the box score.
    fin = pa.groupby(["game_pk", "batting_team"])["runs_pa"].sum().reset_index()
    fin = fin.rename(columns={"runs_pa": "runs_built"})
    truth = pa.groupby("game_pk").agg(
        home_team=("home_team", "first"), away_team=("away_team", "first")).reset_index()
    last_rows = pa.sort_values("at_bat_number").groupby("game_pk").tail(1)
    # Final score per side = max score that side reached.
    side_max = pa.groupby(["game_pk", "batting_team"])["post_bat_score"].max().reset_index()
    side_max = side_max.rename(columns={"post_bat_score": "runs_final"})
    chk = fin.merge(side_max, on=["game_pk", "batting_team"])
    chk["diff"] = chk["runs_built"] - chk["runs_final"]
    bad = chk[chk["diff"].abs() > 0.001]
    pct = 100 * len(bad) / max(len(chk), 1)
    print(f"score reconstruction: {len(chk) - len(bad):,}/{len(chk):,} team-games exact "
          f"({pct:.2f}% mismatched)")
    if pct > 1.0:
        ok = False

    # 2. Every completed team-game should use all nine lineup slots, and slot
    # counts within a game must differ by at most one.
    slots = pa[pa["is_pa"]].groupby(["game_pk", "batting_team"])["lineup_slot"].nunique()
    print(f"team-games using all 9 slots: {(slots == 9).mean() * 100:.2f}% "
          f"(n={len(slots):,})")
    if (slots == 9).mean() < 0.98:
        ok = False

    # 3. Outs per half-inning: completed halves record exactly three.
    halves = pa.groupby("half_key")["outs_pa"].sum()
    print(f"half-innings with exactly 3 outs: {(halves == 3).mean() * 100:.2f}% "
          f"(n={len(halves):,}) -- walk-offs and final halves legitimately differ")

    # 4. Outcome distribution sanity against known league rates.
    d = pa.loc[pa["is_pa"] & pa["outcome"].ne("OTHER"), "outcome"]
    rates = d.value_counts(normalize=True).reindex(config.OUTCOMES).fillna(0)
    print("outcome rates:")
    for k in config.OUTCOMES:
        print(f"  {k:<4} {rates[k] * 100:6.2f}%")
    if not (0.20 <= rates["K"] <= 0.26):
        print("  WARNING: strikeout rate outside plausible league range")
        ok = False

    # 5. Runs per team-game.
    rpg = pa.groupby(["game_pk", "batting_team"])["runs_pa"].sum().mean()
    print(f"runs per team-game: {rpg:.3f}  (league is ~4.3-4.6)")
    if not (4.0 <= rpg <= 5.0):
        ok = False

    print(f"--- {'PASS' if ok else 'FAIL'} ---\n")
    return ok


def main():
    seasons = [int(a) for a in sys.argv[1:]] or None
    raw = load_raw(seasons)
    pa = build(raw)

    # Retain anything that moved the game state, not just completed plate
    # appearances. A handful of at-bats per season end with no event recorded
    # (the half-inning closes on a baserunning out mid-at-bat) and a run can
    # cross during them; dropping those rows loses the run.
    keep_flag = (pa["is_pa"] | pa["raw_event"].ne("none")
                 | pa["runs_pa"].ne(0) | pa["outs_pa"].fillna(0).ne(0))
    pa = pa[keep_flag].copy()

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out = os.path.join(config.PROCESSED_DIR, "pa_events.parquet")
    pa.to_parquet(out, index=False, compression="zstd")
    print(f"wrote {len(pa):,} rows -> {out}")
    print(f"  completed plate appearances: {pa['is_pa'].sum():,}")
    print(f"  games: {pa['game_pk'].nunique():,}   seasons: "
          f"{sorted(pa['season'].unique().tolist())}")

    validate(pa)


if __name__ == "__main__":
    main()
