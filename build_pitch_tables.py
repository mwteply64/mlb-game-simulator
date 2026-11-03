"""Pitch-level aggregates: what a pitcher throws, and what a batter can't hit.

The PA-level model knows a pitcher's overall strikeout rate and a batter's
overall strikeout rate, and multiplies them through a league denominator. What
it cannot see is *how* the strikeouts are produced. Two pitchers with identical
K rates -- one a slider-heavy right-hander, one a changeup-first lefty -- are
interchangeable to it. They are not interchangeable to a batter who cannot
touch a slider.

This builds the tables needed to test whether that distinction carries
information: a pitcher's arsenal mix, and each hitter's outcome profile split
by the kind of pitch that ended the plate appearance.

Outcomes are joined from `pa_events`, never re-derived, so the pitch layer
cannot silently disagree with the PA layer about what happened. The only thing
taken from the raw pitch files is pitch identity and shape.

    python build_pitch_tables.py

Writes pitch_pitcher_mix / pitch_pitcher_group / pitch_batter_group to
data/processed.
"""

import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

# Four groups rather than the usual three. Sinkers and cutters are fastballs by
# velocity but behave nothing like a four-seamer for strikeouts -- a sinker is a
# ground-ball pitch and lumping it with the four-seam hides exactly the contrast
# this is meant to measure.
PITCH_GROUP = {
    "FF": "FF", "FA": "FF",
    "SI": "SI", "FC": "SI",
    "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR", "SV": "BR", "CS": "BR",
    "SC": "BR",
    "CH": "OS", "FS": "OS", "FO": "OS", "EP": "OS",
}
GROUPS = ["FF", "SI", "BR", "OS"]

SHAPE_COLS = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate",
              "arm_angle", "plate_x", "plate_z"]


def load_raw():
    """Pitch identity and shape, plus the keys needed to find each PA's last pitch."""
    keep = (["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter",
             "pitch_type", "p_throws", "stand", "description", "zone",
             "balls", "strikes"] + SHAPE_COLS)
    frames = []
    for f in sorted(glob.glob(os.path.join(config.RAW_DIR, "statcast_*.parquet"))):
        d = pd.read_parquet(f)
        cols = [c for c in keep if c in d.columns]
        frames.append(d[cols])
    out = pd.concat(frames, ignore_index=True)
    out["group"] = out["pitch_type"].map(PITCH_GROUP)
    return out.dropna(subset=["group", "game_pk", "at_bat_number"])


def main():
    print("loading raw pitches...")
    px = load_raw()
    print(f"  {len(px):,} classified pitches")

    pa = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "pa_events.parquet"))
    pa = pa[pa["is_pa"]][["game_pk", "at_bat_number", "season", "batter",
                          "pitcher", "stand", "p_throws", "outcome"]]

    # -- arsenal mix: every pitch counts, not just the ones that ended a PA ---
    px = px.merge(pa[["game_pk", "at_bat_number", "season"]],
                  on=["game_pk", "at_bat_number"], how="inner")

    agg = {c: "mean" for c in SHAPE_COLS if c in px.columns}
    agg["pitch_number"] = "size"
    mix = (px.groupby(["pitcher", "season", "group"]).agg(**{
        **{c: (c, "mean") for c in SHAPE_COLS if c in px.columns},
        "n_pitches": ("pitch_number", "size")}).reset_index())
    mix["share"] = mix["n_pitches"] / mix.groupby(
        ["pitcher", "season"])["n_pitches"].transform("sum")
    mix.to_parquet(os.path.join(config.PROCESSED_DIR,
                                "pitch_pitcher_mix.parquet"), index=False)
    print(f"  arsenal mix: {len(mix):,} pitcher-season-group rows")

    # -- the pitch that ended each PA ---------------------------------------
    # drop_duplicates, not groupby().last(): groupby's per-column last-non-null
    # would stitch together a pitch that never existed.
    last = (px.sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .drop_duplicates(subset=["game_pk", "at_bat_number"], keep="last"))
    ended = pa.merge(last[["game_pk", "at_bat_number", "group"]],
                     on=["game_pk", "at_bat_number"], how="inner")
    print(f"  {len(ended):,} PAs matched to a finishing pitch "
          f"({len(ended) / len(pa) * 100:.1f}% of PAs)")

    for o in config.OUTCOMES:
        ended[o] = (ended["outcome"] == o).astype(np.int32)

    for role, col, name in [("pitcher", "pitcher", "pitch_pitcher_group"),
                            ("batter", "batter", "pitch_batter_group")]:
        t = (ended.groupby([col, "season", "group"], as_index=False)
             [config.OUTCOMES].sum())
        t["pa"] = t[config.OUTCOMES].sum(axis=1)
        t = t.rename(columns={col: "player_id"})
        t.to_parquet(os.path.join(config.PROCESSED_DIR, f"{name}.parquet"),
                     index=False)
        print(f"  {name}: {len(t):,} rows")

    # -- whiffs: the count-independent version of the same question ---------
    # Outcome rates by *finishing* pitch group are badly confounded: breaking
    # balls finish PAs at twice the strikeout rate of sinkers largely because
    # they are thrown in two-strike counts, not because they are twice as
    # nasty. Whiff-per-swing is measured on every pitch, so it survives that
    # objection -- it asks whether this batter misses this kind of pitch, not
    # whether this kind of pitch tends to arrive when he is already behind.
    sw = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
          "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
    wh = {"swinging_strike", "swinging_strike_blocked", "missed_bunt",
          "swinging_pitchout"}
    px["swing"] = px["description"].isin(sw).astype(np.int32)
    px["whiff"] = px["description"].isin(wh).astype(np.int32)
    px["in_zone"] = (pd.to_numeric(px["zone"], errors="coerce") <= 9).astype(np.int32)
    px["two_strike"] = (pd.to_numeric(px["strikes"], errors="coerce") == 2).astype(np.int32)

    for col, name in [("pitcher", "pitch_pitcher_whiff"),
                      ("batter", "pitch_batter_whiff")]:
        t = (px.groupby([col, "season", "group"], as_index=False)
             .agg(pitches=("swing", "size"), swings=("swing", "sum"),
                  whiffs=("whiff", "sum"), in_zone=("in_zone", "sum"),
                  two_strike=("two_strike", "sum")))
        t = t.rename(columns={col: "player_id"})
        t.to_parquet(os.path.join(config.PROCESSED_DIR, f"{name}.parquet"),
                     index=False)
        print(f"  {name}: {len(t):,} rows")

    lgw = px.groupby("group").agg(sw=("swing", "sum"), wh=("whiff", "sum"),
                                  n=("swing", "size"))
    print("\nleague whiff rate per swing by pitch group (count-independent):")
    for g in GROUPS:
        if g in lgw.index:
            print(f"  {g}  whiff/swing {lgw.at[g, 'wh'] / lgw.at[g, 'sw'] * 100:5.1f}%"
                  f"   share of all pitches {lgw.at[g, 'n'] / lgw['n'].sum() * 100:5.1f}%")

    # -- sanity: the contrast this is built to exploit should be visible -----
    lg = (ended.groupby("group")[config.OUTCOMES].sum())
    lg = lg.div(lg.sum(axis=1), axis=0)
    print("\nleague outcome rate by finishing pitch group:")
    for g in GROUPS:
        if g in lg.index:
            print(f"  {g}  K {lg.at[g, 'K'] * 100:5.1f}%   "
                  f"BB {lg.at[g, 'BB'] * 100:4.1f}%   "
                  f"HR {lg.at[g, 'HR'] * 100:4.1f}%   "
                  f"share of PAs {(ended['group'] == g).mean() * 100:4.1f}%")


if __name__ == "__main__":
    main()
