"""As-of projections for individual defense and baserunning ratings.

Both are projected the same way the outcome model projects hitting: prior
seasons, decayed for recency, regressed toward league average by how much
exposure the player actually has. Neither metric is used raw -- single-season
fielding numbers in particular are far noisier than they look, with year-to-year
correlation around 0.4 to 0.5 even for full-time players.

**Defense** is expressed as runs prevented per defensive plate appearance, so
the nine players on the field at any moment sum to a team defensive rate. That
sum is what shifts balls in play between hits and outs in the simulator.

**Baserunning** is sprint speed, converted to a league percentile so it can
bucket the base-out transition tables.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import fetch_player_ratings as fpr

# Recency decay per season, matching the outcome model's treatment.
DECAY_DEFENSE = 0.80
DECAY_SPEED = 0.85

# Regression constant for defense, in defensive plate appearances. A full-time
# fielder is on the field for roughly 5,000 of his team's defensive plate
# appearances in a season, so 6,000 pulls a single season a little past halfway
# toward league average -- consistent with how slowly fielding metrics
# stabilize.
REGRESS_PA_DEFENSE = 6000

# Regression constant for sprint speed, in competitive runs. Speed is a
# physical measurement and stabilizes fast, so this is small.
REGRESS_RUNS_SPEED = 30

LEAGUE_SPEED_FALLBACK = 27.2


def load_ratings():
    oaa = (pd.read_parquet(fpr.OAA_PATH)
           if os.path.exists(fpr.OAA_PATH) else pd.DataFrame())
    speed = (pd.read_parquet(fpr.SPEED_PATH)
             if os.path.exists(fpr.SPEED_PATH) else pd.DataFrame())
    return oaa, speed


def defensive_exposure(pa):
    """Count the defensive plate appearances each player was on the field for.

    The OAA feed reports season run totals but not opportunities, so exposure is
    counted here from the actual alignment on every plate appearance. This is
    what converts a counting stat into a rate that can be summed across a
    lineup.
    """
    cols = [f"fielder_{i}" for i in range(2, 10)]
    have = [c for c in cols if c in pa.columns]
    if not have:
        raise SystemExit(
            "fielder_* columns absent from pa_events. Re-run fetch_statcast.py "
            "--force after adding them to config.STATCAST_COLUMNS, then rebuild.")

    d = pa[pa["is_pa"]]
    frames = []
    for c in have:
        f = d[["season", c]].rename(columns={c: "player_id"}).dropna()
        frames.append(f)
    stacked = pd.concat(frames, ignore_index=True)
    stacked["player_id"] = stacked["player_id"].astype(int)
    return (stacked.groupby(["season", "player_id"]).size()
            .rename("def_pa").reset_index())


class DefenseRatings:
    """Projected runs prevented per defensive plate appearance, by player."""

    def __init__(self, oaa, exposure, asof_season):
        hist = oaa[oaa["season"] < asof_season].merge(
            exposure, on=["season", "player_id"], how="left")
        hist["def_pa"] = hist["def_pa"].fillna(0.0)
        hist["weight"] = DECAY_DEFENSE ** (asof_season - hist["season"] - 1)

        hist["runs"] = hist["fielding_runs_prevented"].fillna(0.0) * hist["weight"]
        hist["exposure"] = hist["def_pa"] * hist["weight"]

        g = hist.groupby("player_id")[["runs", "exposure"]].sum()
        # Regress toward zero, which is league average by construction.
        rate = g["runs"] / (g["exposure"] + REGRESS_PA_DEFENSE)
        self.rate = rate.to_dict()
        self.exposure = g["exposure"].to_dict()

    def get(self, player_id):
        return self.rate.get(int(player_id), 0.0)

    def team_rate(self, fielder_ids):
        """Summed rate across whoever is on the field."""
        return float(sum(self.get(p) for p in fielder_ids if p and p == p))


class SpeedRatings:
    """Projected sprint speed and its league percentile, by player."""

    def __init__(self, speed, asof_season):
        hist = speed[speed["season"] < asof_season].copy()
        if len(hist) == 0:
            hist = speed.copy()
        hist["weight"] = DECAY_SPEED ** (asof_season - hist["season"] - 1)

        recent = hist[hist["season"] >= asof_season - 3]
        self.league_mean = float(recent["sprint_speed"].mean()) \
            if len(recent) else LEAGUE_SPEED_FALLBACK

        hist["runs_w"] = hist["competitive_runs"].fillna(0.0) * hist["weight"]
        hist["speed_w"] = hist["sprint_speed"] * hist["runs_w"]

        g = hist.groupby("player_id")[["runs_w", "speed_w"]].sum()
        raw = g["speed_w"] / g["runs_w"].replace(0, np.nan)
        shrunk = ((g["speed_w"] + REGRESS_RUNS_SPEED * self.league_mean)
                  / (g["runs_w"] + REGRESS_RUNS_SPEED))
        self.speed = shrunk.where(raw.notna(), self.league_mean).to_dict()

        vals = np.array(sorted(self.speed.values()))
        self._sorted = vals

    def get(self, player_id):
        return self.speed.get(int(player_id), self.league_mean)

    def percentile(self, player_id):
        v = self.get(player_id)
        if len(self._sorted) == 0:
            return 0.5
        return float(np.searchsorted(self._sorted, v) / len(self._sorted))

    def bucket(self, player_id, n_buckets=3):
        """Speed tercile: 0 slow, 1 average, 2 fast."""
        p = self.percentile(player_id)
        return min(int(p * n_buckets), n_buckets - 1)


def defense_hit_suppression(team_rate, base_probs, lw):
    """Fraction of balls-in-play hits a defense converts into outs.

    A team defense worth `team_rate` runs per plate appearance has to move
    enough probability from singles, doubles, and triples into outs to change
    expected run value by exactly that much. Home runs are excluded -- no
    fielder catches those -- and strikeouts, walks, and hit batsmen are
    untouched, since range does not act on them.

    Returns the fraction `h` to remove from each ball-in-play hit category.
    """
    i1, i2, i3 = (config.OUTCOMES.index(o) for o in ("1B", "2B", "3B"))
    i_out = config.OUTCOMES.index("OUT")

    p_hits = base_probs[[i1, i2, i3]]
    hit_value = float(np.dot(p_hits, lw[[i1, i2, i3]]))
    out_value = float(p_hits.sum() * lw[i_out])

    denom = hit_value - out_value
    if abs(denom) < 1e-9:
        return 0.0
    # Positive team_rate means runs prevented, so hits must go down.
    return float(np.clip(team_rate / denom, -0.5, 0.5))


def apply_defense(probs, h):
    """Move a fraction `h` of ball-in-play hits into outs."""
    i1, i2, i3 = (config.OUTCOMES.index(o) for o in ("1B", "2B", "3B"))
    i_out = config.OUTCOMES.index("OUT")

    v = probs.copy()
    moved = 0.0
    for i in (i1, i2, i3):
        delta = v[i] * h
        v[i] -= delta
        moved += delta
    v[i_out] += moved
    return v / v.sum()


def main():
    import outcome_model as om

    pa = om.load_pa()
    oaa, speed = load_ratings()
    exposure = defensive_exposure(pa)

    print("defensive exposure (plate appearances on the field):")
    print(exposure.groupby("season")["def_pa"].agg(["count", "mean", "max"])
          .round(0).to_string())

    dr = DefenseRatings(oaa, exposure, 2025)
    rates = pd.Series(dr.rate).sort_values()
    print(f"\nprojected defense as of 2025: {len(rates):,} players")
    print(f"  runs prevented per defensive PA, range "
          f"{rates.min():+.5f} to {rates.max():+.5f}")
    print(f"  implied over a 6,100-PA team season: "
          f"{rates.min() * 6100:+.1f} to {rates.max() * 6100:+.1f} runs "
          "for a single player")

    sr = SpeedRatings(speed, 2025)
    sp = pd.Series(sr.speed)
    print(f"\nprojected sprint speed as of 2025: {len(sp):,} players")
    print(f"  league mean {sr.league_mean:.2f} ft/s, "
          f"range {sp.min():.2f} to {sp.max():.2f}")


if __name__ == "__main__":
    main()
