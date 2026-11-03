"""Shared paths, season scope, and outcome taxonomy for the game simulator."""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# Sibling project we borrow projection methodology (not data files) from.
LINEUP_PROJECT_DIR = os.path.join(
    os.path.dirname(PROJECT_DIR), "lineup optimization project"
)

# 2020 is excluded (60-game COVID season). 2021 is the earliest year whose
# bullpen behavior is comparable to today's: the three-batter minimum arrived
# in 2020 and killed the single-batter lefty specialist. Note that the batted
# ball environment shifts again in 2023 (shift restrictions, pitch clock), so
# tactical models should be trained on 2023+ even though the plate-appearance
# outcome model can use the full span.
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
TACTICAL_SEASONS = [2023, 2024, 2025, 2026]

# Generous bounds; game_type filtering discards spring training and postseason.
SEASON_START_MMDD = "03-15"
SEASON_END_MMDD = "10-10"

# The eight mutually exclusive plate appearance outcomes the simulator draws
# from. Everything Statcast reports collapses into exactly one of these.
OUTCOMES = ["K", "BB", "HBP", "OUT", "1B", "2B", "3B", "HR"]

# Columns retained from Statcast's 119. Keeps the raw cache to a few hundred MB
# instead of several GB while preserving everything the later stages need.
STATCAST_COLUMNS = [
    # identity / ordering
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "home_team", "away_team",
    # participants
    "batter", "pitcher", "stand", "p_throws",
    # defensive alignment -- who is actually on the field, by position number.
    # Needed to aggregate individual fielding ratings into the team defense
    # that is actually behind the pitcher on any given plate appearance.
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    # game state as of this pitch
    "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "balls", "strikes", "bat_score", "fld_score", "post_bat_score",
    "home_score", "away_score", "post_home_score", "post_away_score",
    # outcome
    "events", "description", "des", "type",
    # batted ball detail
    "bb_type", "launch_speed", "launch_angle", "hit_location",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "babip_value", "iso_value",
    # context features for later stages
    "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game",
    "if_fielding_alignment", "of_fielding_alignment",
    "delta_home_win_exp", "delta_run_exp",
    "pitch_type", "release_speed",

    # --- pitch characteristics, for arsenal and location matchups ----------
    # All of these carry 99%+ coverage back to 2021, so they can train models
    # that have to project earlier seasons.
    #
    # Location matters beyond pitch type: a slider away from a left-handed bat
    # and a slider in are different pitches with opposite platoon implications,
    # and `api_break_x_batter_in` measures break toward or away from the batter
    # rather than in absolute field coordinates, which is the frame that
    # actually matters for a matchup.
    "plate_x", "plate_z", "zone", "sz_top", "sz_bot",
    "pfx_x", "pfx_z", "api_break_z_with_gravity", "api_break_x_arm",
    "api_break_x_batter_in",
    "arm_angle", "release_pos_x", "release_pos_z", "release_extension",
    "release_spin_rate", "spin_axis", "effective_speed",

    # --- bat tracking: 2025 ONWARD ONLY, and only on swings ---------------
    # Zero coverage in 2021 and 2023, roughly 48% in 2025-26 (a swing has to
    # happen). Usable for current-season projection and live predictions;
    # cannot train anything that must backtest earlier seasons.
    "bat_speed", "swing_length", "attack_angle", "swing_path_tilt",
]


def season_bounds(year):
    return f"{year}-{SEASON_START_MMDD}", f"{year}-{SEASON_END_MMDD}"


def raw_chunk_path(year, chunk_index):
    return os.path.join(RAW_DIR, f"statcast_{year}_{chunk_index:02d}.parquet")
