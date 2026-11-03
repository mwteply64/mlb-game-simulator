"""Plate appearance outcome model.

Given a batter, a pitcher, and context, produce a probability vector over the
eight outcomes the simulator draws from. Three ideas do the work:

**Per-outcome shrinkage.** Every rate is regressed toward league average by an
amount specific to that outcome. Strikeout rate is nearly a batter's own after
60 plate appearances; doubles rate is still mostly league average after 600.
Using one regression constant for all outcomes -- or none -- is the single
biggest source of overconfidence in homemade projections.

**Odds-ratio combination.** Batter and pitcher rates are combined as
b*p/league, then renormalized. This is Tango's multinomial odds ratio: it has
the right fixed points (a league-average pitcher returns the batter's own
rates, and vice versa) which naive averaging does not.

**Evidence-weighted platoon.** Rather than estimating vs-LHP rates from a
hitter's thin vs-LHP sample, the model applies his overall rates times a
platoon ratio that is itself blended between his own observed ratio and a
league prior, weighted by how much vs-hand evidence he actually has. This is
the structure that shipped as v16 of the lineup optimizer, where it beat
estimating the split directly.

Everything is computed as-of a date, so no plate appearance is ever projected
using information from itself or later.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)
OUT_IDX = {o: i for i, o in enumerate(OUTCOMES)}

# --------------------------------------------------------------------------
# Tunable constants. Starting values come from published stabilization points
# (Carleton) converted to regression-toward-mean constants; they are the first
# thing a tuner should sweep, jointly, on held-out seasons.
# --------------------------------------------------------------------------

# Regression constants in plate appearances, per outcome, batter side.
REGRESS_PA_BATTER = {
    "K": 60, "BB": 120, "HBP": 240, "OUT": 70,
    "1B": 290, "2B": 500, "3B": 1200, "HR": 170,
}

# Pitchers stabilize faster than hitters on the things they control (strikeouts,
# walks) and slower on batted-ball results, which is the whole FIP insight.
REGRESS_PA_PITCHER = {
    "K": 70, "BB": 170, "HBP": 640, "OUT": 90,
    "1B": 670, "2B": 1200, "3B": 2000, "HR": 1300,
}

# Exponential decay per season of distance from the as-of date, applied to
# career counts. 1.0 = no decay.
DECAY_PER_SEASON_BATTER = 0.85
DECAY_PER_SEASON_PITCHER = 0.80

# Evidence weight for the platoon ratio: own observed ratio gets weight
# n_vs_hand / (n_vs_hand + PLATOON_PRIOR_PA).
PLATOON_PRIOR_PA_BATTER = 400
PLATOON_PRIOR_PA_PITCHER = 500

# Minimum counts before a player is treated as having any signal at all.
MIN_PA_FOR_OWN_RATES = 25

# Regression constant, in plate appearances, for nowcasting the league
# environment. The league baseline drifts year to year -- 2025 ran 11% below
# the 2023-24 mean on triples -- so a trailing multi-season prior is stale by
# construction and pushes that error into every projection. Season-to-date
# league rates are legitimately available at prediction time, so they are
# blended in: at 30,000 plate appearances (roughly six weeks) the current
# season carries half the weight.
LEAGUE_NOWCAST_PA = 30000

# Optional run-wide constant overrides, set by a caller that wants to evaluate
# a tuned set. None means use the module defaults above.
ACTIVE_PARAMS = None


# --------------------------------------------------------------------------
# Count tables
# --------------------------------------------------------------------------

def _outcome_dummies(pa):
    d = pd.get_dummies(pa["outcome"])
    for o in OUTCOMES:
        if o not in d.columns:
            d[o] = 0
    return d[OUTCOMES].astype(np.int32)


def build_count_tables(pa):
    """Aggregate plate appearances to (player, season, opposing hand) counts.

    Keeping season as a dimension lets any as-of window be assembled by
    filtering and summing, with recency decay applied at assembly time.
    """
    pa = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    dummies = _outcome_dummies(pa)
    pa = pd.concat([pa.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    bat = pa.groupby(["batter", "season", "p_throws"], as_index=False)[OUTCOMES].sum()
    bat = bat.rename(columns={"batter": "player_id", "p_throws": "opp_hand"})
    bat["pa"] = bat[OUTCOMES].sum(axis=1)

    pit = pa.groupby(["pitcher", "season", "stand"], as_index=False)[OUTCOMES].sum()
    pit = pit.rename(columns={"pitcher": "player_id", "stand": "opp_hand"})
    pit["pa"] = pit[OUTCOMES].sum(axis=1)

    lg = pa.groupby("season", as_index=False)[OUTCOMES].sum()
    lg["pa"] = lg[OUTCOMES].sum(axis=1)

    lg_split = pa.groupby(["season", "stand", "p_throws"], as_index=False)[OUTCOMES].sum()
    lg_split["pa"] = lg_split[OUTCOMES].sum(axis=1)

    return {"batter": bat, "pitcher": pit, "league": lg, "league_split": lg_split}


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------

class OutcomeModel:
    """Projects plate appearance outcome vectors as of a given season.

    `asof_season` is the season being predicted; only strictly prior seasons
    plus in-season data supplied through `season_to_date` are used.
    """

    def __init__(self, tables, asof_season, season_to_date=None, params=None):
        """`season_to_date` is None, or a dict with "batter" and "pitcher"
        count tables covering the part of `asof_season` already played. The two
        sides must be passed separately -- feeding one table to both builders
        mixes batter rows into the pitcher aggregation.

        `params` overrides the module-level constants, so a tuner can evaluate
        candidate values without mutating global state (which would leak
        between folds and make results depend on evaluation order).
        """
        # ACTIVE_PARAMS lets a whole run adopt a tuned constant set without
        # threading it through every construction site. The tuner never uses
        # it -- it passes `params` explicitly, so folds cannot leak into each
        # other or depend on evaluation order.
        p = params if params is not None else (ACTIVE_PARAMS or {})
        self.regress_batter = p.get("REGRESS_PA_BATTER", REGRESS_PA_BATTER)
        self.regress_pitcher = p.get("REGRESS_PA_PITCHER", REGRESS_PA_PITCHER)
        self.decay_batter = p.get("DECAY_PER_SEASON_BATTER", DECAY_PER_SEASON_BATTER)
        self.decay_pitcher = p.get("DECAY_PER_SEASON_PITCHER", DECAY_PER_SEASON_PITCHER)
        self.platoon_prior_batter = p.get("PLATOON_PRIOR_PA_BATTER",
                                          PLATOON_PRIOR_PA_BATTER)
        self.platoon_prior_pitcher = p.get("PLATOON_PRIOR_PA_PITCHER",
                                           PLATOON_PRIOR_PA_PITCHER)
        self.league_nowcast_pa = p.get("LEAGUE_NOWCAST_PA", LEAGUE_NOWCAST_PA)
        self.min_pa_own = p.get("MIN_PA_FOR_OWN_RATES", MIN_PA_FOR_OWN_RATES)

        self.asof_season = asof_season
        self.tables = tables
        std = season_to_date or {}
        if not isinstance(std, dict):
            raise TypeError("season_to_date must be None or a dict with "
                            "'batter' and 'pitcher' count tables")

        self.league = self._league_vector(tables["league"], asof_season,
                                          std.get("league"))
        self.league_cell = self._league_cells(tables["league_split"], asof_season,
                                              std.get("league_split"))
        self.platoon_prior = self._league_platoon_ratios(tables["league_split"], asof_season)

        self.bat_rates, self.bat_pa = self._player_rates(
            tables["batter"], asof_season, std.get("batter"), "batter",
            self.regress_batter, self.decay_batter)
        self.pit_rates, self.pit_pa = self._player_rates(
            tables["pitcher"], asof_season, std.get("pitcher"), "pitcher",
            self.regress_pitcher, self.decay_pitcher)

    # -- league ------------------------------------------------------------

    def _nowcast(self, prior_rate, std_counts):
        """Blend season-to-date league counts onto a prior-seasons rate."""
        if std_counts is None:
            return prior_rate
        n = float(std_counts.sum())
        if n <= 0:
            return prior_rate
        k = self.league_nowcast_pa
        v = (std_counts + k * prior_rate) / (n + k)
        return v / v.sum()

    def _league_vector(self, lg, asof, std_league=None):
        prior = lg[lg["season"] < asof]
        if len(prior) == 0:
            prior = lg
        recent = prior.sort_values("season").tail(2)
        counts = recent[OUTCOMES].sum().to_numpy(dtype=float)
        prior_rate = counts / counts.sum()

        std_counts = None
        if std_league is not None and len(std_league):
            rows = std_league[std_league["season"] == asof]
            if len(rows):
                std_counts = rows[OUTCOMES].sum().to_numpy(dtype=float)
        return self._nowcast(prior_rate, std_counts)

    def _league_cells(self, lg_split, asof, std_split=None):
        """League outcome rates within each (batter side, pitcher hand) cell.

        This is the correct denominator for the odds ratio once both sides have
        been platoon-adjusted. Dividing by the hand-agnostic league vector
        instead counts the platoon effect twice -- once from the hitter's
        vs-hand ratio and again from the pitcher's vs-side ratio -- which
        inflates every left-on-left and right-on-right matchup.
        """
        prior = lg_split[lg_split["season"] < asof]
        if len(prior) == 0:
            prior = lg_split

        std_rows = None
        if std_split is not None and len(std_split):
            std_rows = std_split[std_split["season"] == asof]

        cells = {}
        for (side, hand), grp in prior.groupby(["stand", "p_throws"]):
            v = grp[OUTCOMES].sum().to_numpy(dtype=float)
            if v.sum() <= 0:
                continue
            prior_rate = v / v.sum()

            std_counts = None
            if std_rows is not None:
                cell = std_rows[(std_rows["stand"] == side)
                                & (std_rows["p_throws"] == hand)]
                if len(cell):
                    std_counts = cell[OUTCOMES].sum().to_numpy(dtype=float)
            cells[(side, hand)] = self._nowcast(prior_rate, std_counts)
        return cells

    def denominator(self, stand, p_throws, use_platoon=True):
        if use_platoon:
            return self.league_cell.get((stand, p_throws), self.league)
        return self.league

    def _league_platoon_ratios(self, lg_split, asof):
        """Ratio of each outcome rate vs one hand to the batter's overall rate,
        by (batter side, pitcher hand). The league prior a thin-sample player
        is pulled toward."""
        prior = lg_split[lg_split["season"] < asof]
        if len(prior) == 0:
            prior = lg_split

        out = {}
        for side in ["L", "R"]:
            side_rows = prior[prior["stand"] == side]
            if len(side_rows) == 0:
                continue
            overall = side_rows[OUTCOMES].sum().to_numpy(dtype=float)
            overall = overall / overall.sum()
            for hand in ["L", "R"]:
                cell = side_rows[side_rows["p_throws"] == hand]
                if len(cell) == 0:
                    continue
                v = cell[OUTCOMES].sum().to_numpy(dtype=float)
                v = v / v.sum()
                out[(side, hand)] = np.where(overall > 0, v / np.maximum(overall, 1e-9), 1.0)

        # Pitcher-side priors: same idea, indexed by (pitcher hand, batter side).
        for hand in ["L", "R"]:
            hand_rows = prior[prior["p_throws"] == hand]
            if len(hand_rows) == 0:
                continue
            overall = hand_rows[OUTCOMES].sum().to_numpy(dtype=float)
            overall = overall / overall.sum()
            for side in ["L", "R"]:
                cell = hand_rows[hand_rows["stand"] == side]
                if len(cell) == 0:
                    continue
                v = cell[OUTCOMES].sum().to_numpy(dtype=float)
                v = v / v.sum()
                out[("P", hand, side)] = np.where(overall > 0, v / np.maximum(overall, 1e-9), 1.0)
        return out

    # -- players -----------------------------------------------------------

    def _player_rates(self, tbl, asof, season_to_date, role, regress, decay):
        """Shrunk overall rates plus per-hand counts, for every player.

        Returns (rates_by_pid, pa_by_pid) where rates are the player's overall
        shrunk 8-vector and pa carries both the overall and per-hand sample
        sizes needed by the platoon step.
        """
        hist = tbl[tbl["season"] < asof].copy()
        hist["weight"] = decay ** (asof - hist["season"] - 1)

        if season_to_date is not None and len(season_to_date):
            std = season_to_date[season_to_date["season"] == asof].copy()
            std["weight"] = 1.0
            hist = pd.concat([hist, std], ignore_index=True)

        for o in OUTCOMES:
            hist[o] = hist[o] * hist["weight"]
        hist["pa"] = hist["pa"] * hist["weight"]

        overall = hist.groupby("player_id", as_index=False)[OUTCOMES + ["pa"]].sum()
        by_hand = hist.groupby(["player_id", "opp_hand"], as_index=False)[OUTCOMES + ["pa"]].sum()

        # Each outcome is shrunk as its own binomial, with its own denominator:
        #     rate_o = (x_o + k_o * league_o) / (n + k_o)
        # A shared denominator (n + sum(k)) is wrong -- it over-shrinks every
        # outcome by the total of all the constants and leaves a vector that
        # does not sum to one, so renormalizing then distorts the proportions.
        # With n = 0 this returns league rates exactly; as n grows it converges
        # on the player's own rates, each at its own speed.
        k = np.array([regress[o] for o in OUTCOMES], dtype=float)
        counts = overall[OUTCOMES].to_numpy(dtype=float)
        n = counts.sum(axis=1, keepdims=True)
        shrunk = (counts + k * self.league) / (n + k)
        shrunk = shrunk / shrunk.sum(axis=1, keepdims=True)

        rates = {int(p): shrunk[i] for i, p in enumerate(overall["player_id"])}
        pa_overall = {int(p): float(v) for p, v in
                      zip(overall["player_id"], overall["pa"])}

        hand_counts = {}
        for pid, hand, *vals, npa in by_hand[
                ["player_id", "opp_hand"] + OUTCOMES + ["pa"]].itertuples(index=False):
            hand_counts[(int(pid), hand)] = (np.array(vals, dtype=float), float(npa))

        return rates, {"overall": pa_overall, "by_hand": hand_counts}

    # -- projection --------------------------------------------------------

    def _platoon_adjusted(self, pid, rates, pa_info, prior_key, prior_pa):
        """Player's overall rates scaled by an evidence-weighted platoon ratio."""
        base = rates.get(pid)
        if base is None:
            return self.league.copy()

        league_ratio = self.platoon_prior.get(prior_key)
        if league_ratio is None:
            league_ratio = np.ones(N_OUT)

        # The opposing hand is the last element of the prior key either way:
        # (stand, p_throws) for a batter, ("P", p_throws, stand) for a pitcher.
        opp_hand = prior_key[-1]
        counts, n_hand = pa_info["by_hand"].get((pid, opp_hand), (None, 0.0))
        if counts is not None and n_hand >= self.min_pa_own:
            obs = counts / max(counts.sum(), 1e-9)
            own_ratio = np.where(base > 0, obs / np.maximum(base, 1e-9), 1.0)
            # Clamp before blending: a hitter with 40 plate appearances and two
            # triples must not be handed a 12x triple multiplier.
            own_ratio = np.clip(own_ratio, 0.4, 2.5)
        else:
            own_ratio = league_ratio

        w = n_hand / (n_hand + prior_pa)
        ratio = w * own_ratio + (1.0 - w) * league_ratio

        v = base * ratio
        return v / v.sum()

    def batter_vector(self, batter_id, stand, p_throws):
        return self._platoon_adjusted(
            int(batter_id), self.bat_rates, self.bat_pa,
            (stand, p_throws), self.platoon_prior_batter)

    def pitcher_vector(self, pitcher_id, p_throws, stand):
        return self._platoon_adjusted(
            int(pitcher_id), self.pit_rates, self.pit_pa,
            ("P", p_throws, stand), self.platoon_prior_pitcher)

    def matchup(self, batter_id, pitcher_id, stand, p_throws,
                tto=1, park_mult=None):
        """The odds-ratio combination the simulator calls once per plate
        appearance."""
        b = self.batter_vector(batter_id, stand, p_throws)
        p = self.pitcher_vector(pitcher_id, p_throws, stand)

        lg = self.denominator(stand, p_throws, use_platoon=True)
        v = b * p / np.maximum(lg, 1e-12)
        if park_mult is not None:
            v = v * park_mult
        if tto and tto > 1:
            v = v * TTO_MULT.get(min(int(tto), 4), np.ones(N_OUT))
        return v / v.sum()


# Times-through-the-order multipliers, filled in by fit_tto() and pasted back
# here once estimated. Identity until then.
TTO_MULT = {1: np.ones(N_OUT), 2: np.ones(N_OUT),
            3: np.ones(N_OUT), 4: np.ones(N_OUT)}


def fit_tto(pa, seasons=None):
    """Estimate per-outcome times-through-order multipliers.

    Naive TTO splits are confounded: pitchers who reach a third turn are the
    ones pitching well, and they face the top of the order more often. This
    controls for pitcher identity by measuring each pitcher against his own
    first-time-through rates, then pools.
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    d = d[d["n_thruorder_pitcher"].between(1, 4)]

    dummies = _outcome_dummies(d)
    d = pd.concat([d.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    g = d.groupby(["pitcher", "n_thruorder_pitcher"], as_index=False)[OUTCOMES].sum()
    g["pa"] = g[OUTCOMES].sum(axis=1)

    first = g[g["n_thruorder_pitcher"] == 1].set_index("pitcher")
    out = {1: np.ones(N_OUT)}
    for t in [2, 3, 4]:
        sub = g[g["n_thruorder_pitcher"] == t].set_index("pitcher")
        common = sub.index.intersection(first.index)
        if len(common) == 0:
            out[t] = np.ones(N_OUT)
            continue

        # Weight each pitcher by his smaller sample so a swingman with 12 plate
        # appearances at TTO 3 does not move the estimate as much as an ace.
        w = np.minimum(sub.loc[common, "pa"], first.loc[common, "pa"]).to_numpy()
        num = (sub.loc[common, OUTCOMES].to_numpy()
               / np.maximum(sub.loc[common, "pa"].to_numpy()[:, None], 1))
        den = (first.loc[common, OUTCOMES].to_numpy()
               / np.maximum(first.loc[common, "pa"].to_numpy()[:, None], 1))
        ratio = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)
        ratio = np.clip(ratio, 0.5, 2.0)

        mult = np.average(ratio, axis=0, weights=w)
        # Renormalize against the first-time-through rate vector so applying
        # the multiplier leaves total probability mass at 1 rather than
        # inflating it.
        base = np.average(den, axis=0, weights=w)
        base = base / base.sum()
        out[t] = mult / float(np.dot(base, mult))
    return out


def fit_park_factors(pa, asof_season, n_years=3, regress_games=200,
                     prefer_savant=True):
    """Per-outcome park multipliers.

    Prefers Baseball Savant's published factors when the cache exists. Theirs
    compare each team's home performance to its own road performance, which
    cancels team quality; the fallback below compares a park to the *league*,
    which conflates the two -- a club with poor pitching makes its park look
    friendlier than it is.

    The fallback also over-regressed. Against Savant it understated every
    extreme park: Coors triples 1.55 against their 1.98, Yankee Stadium 0.71
    against 0.51. A 200-game regression constant was far too aggressive for an
    effect this stable.
    """
    if prefer_savant:
        try:
            import fetch_park_factors as fpf
            savant = fpf.load(asof_season)
            if savant:
                return savant
        except Exception:      # noqa: BLE001 - fall back rather than fail
            pass

    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    d = d[d["season"].between(asof_season - n_years, asof_season - 1)]
    if len(d) == 0:
        return {}

    dummies = _outcome_dummies(d)
    d = pd.concat([d.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    lg = d[OUTCOMES].sum().to_numpy(dtype=float)
    lg = lg / lg.sum()

    g = d.groupby("park", as_index=False)[OUTCOMES].sum()
    g["pa"] = g[OUTCOMES].sum(axis=1)
    games = d.groupby("park")["game_pk"].nunique()

    out = {}
    for _, row in g.iterrows():
        v = row[OUTCOMES].to_numpy(dtype=float)
        v = v / v.sum()
        raw = np.where(lg > 0, v / np.maximum(lg, 1e-9), 1.0)
        ng = float(games.get(row["park"], 0))
        w = ng / (ng + regress_games)
        out[row["park"]] = w * raw + (1.0 - w) * 1.0
    return out


def fit_home_advantage(pa, asof_season, n_years=4):
    """Per-outcome home/away multipliers.

    Home field advantage is a real, sizeable per-plate-appearance effect --
    home batters strike out ~3.6% less, walk ~6.8% more, and homer ~4.6% more
    than away batters. It is not a park effect (both teams bat in the same
    park, so park factors cancel) and not a schedule artifact (home and away
    assignments balance over a season).

    Comparing runs *per game* instead would hide it almost entirely: home teams
    skip the bottom of the ninth when they are ahead, so they bat roughly half
    an inning less in about 45% of games and their runs-per-game looks flat
    even though their rate of production is higher.

    Returned multipliers are relative to the pooled league rate, so applying
    them to a league-average hitter reproduces the observed home and away rates
    exactly and leaves the overall league environment unchanged.
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)]
    d = d[d["season"].between(asof_season - n_years, asof_season - 1)]
    if len(d) == 0:
        return {"home": np.ones(N_OUT), "away": np.ones(N_OUT)}

    pooled = d["outcome"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    out = {}
    for key, mask in [("home", d["batting_is_home"]), ("away", ~d["batting_is_home"])]:
        v = (d.loc[mask, "outcome"].value_counts(normalize=True)
             .reindex(OUTCOMES).fillna(0).to_numpy())
        out[key] = np.where(pooled > 0, v / np.maximum(pooled, 1e-12), 1.0)
    return out


def fit_calibration(tables, pa, calib_season, n_iter=4, park_factors=None):
    """Per-outcome multiplicative correction for odds-ratio normalization bias.

    The multinomial odds ratio is not mean-preserving. Each matchup is divided
    by its own normalizer S = sum(b*p/l), which varies across matchups. For a
    rare outcome, its own contribution to S is negligible, so the outcome and
    the normalizer are near-independent and E[X/S] ~ E[X]*E[1/S] > l -- rare
    outcomes come out systematically inflated. Common outcomes are effectively
    self-normalizing and get pushed down to compensate. Measured on 2025 this
    was +24% on triples and -2.4% on strikeouts.

    The target is the model's own league vector, NOT the season's observed
    outcomes. Averaged over a full season every plate appearance has one batter
    and one pitcher drawn from the league, so the mean prediction should
    reproduce league rates exactly; whatever it misses by is the distortion.
    Fitting against observed outcomes instead folds that season's environment
    drift into the correction -- fit on 2024 that way, the correction absorbed
    2024's low home run rate and then over-suppressed home runs in 2025.

    Iterated, because each pass changes the normalizer and one pass undershoots.
    """
    model = OutcomeModel(tables, calib_season, season_to_date=None)
    d = pa[(pa["season"] == calib_season) & pa["is_pa"]
           & pa["outcome"].isin(OUTCOMES)]
    if len(d) == 0:
        return np.ones(N_OUT)

    bat = d["batter"].to_numpy()
    pit = d["pitcher"].to_numpy()
    stands = d["stand"].to_numpy()
    hands = d["p_throws"].to_numpy()
    parks = d["park"].to_numpy()
    tto = d["n_thruorder_pitcher"].fillna(1).clip(1, 4).to_numpy().astype(int)

    bcache, pcache = {}, {}
    n = len(d)
    B = np.empty((n, N_OUT))
    P = np.empty((n, N_OUT))
    D = np.empty((n, N_OUT))
    for i in range(n):
        s, h = stands[i], hands[i]
        kb = (int(bat[i]), s, h)
        if kb not in bcache:
            bcache[kb] = model.batter_vector(int(bat[i]), s, h)
        kp = (int(pit[i]), h, s)
        if kp not in pcache:
            pcache[kp] = model.pitcher_vector(int(pit[i]), h, s)
        B[i] = bcache[kb]
        P[i] = pcache[kp]
        D[i] = model.denominator(s, h, use_platoon=True)

    base = B * P / np.maximum(D, 1e-12)
    base = base * np.stack([TTO_MULT.get(t, np.ones(N_OUT)) for t in tto])
    if park_factors:
        base = base * np.stack([park_factors.get(p, np.ones(N_OUT)) for p in parks])

    target = model.league

    calib = np.ones(N_OUT)
    for _ in range(n_iter):
        V = base * calib
        V = V / V.sum(axis=1, keepdims=True)
        pred = V.mean(axis=0)
        calib = calib * (target / np.maximum(pred, 1e-12))
        calib = calib / calib.sum() * N_OUT  # keep the scale from drifting
    return calib


def load_pa():
    path = os.path.join(config.PROCESSED_DIR, "pa_events.parquet")
    return pd.read_parquet(path)
