"""Vectorized Monte Carlo game simulation.

Every simulation of a game advances in lockstep, one plate appearance per step,
with state held in parallel NumPy arrays. Simulations sitting in different
innings or half-innings is fine -- state is per-simulation -- so the whole
batch steps together and the inner loop is array math rather than Python.
That is what makes tens of millions of simulated plate appearances practical
without dropping to numba or C.

This is the step-two baseline: **fixed pitcher usage**. The starter is pulled
at a batters-faced threshold and generic league-average relievers finish. The
hook and bullpen-selection models replace `HookRule` in step three, and the
delta between this baseline and that version is the honest measure of what the
bullpen layer is worth.
"""

import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np

import config
from player_ratings import apply_defense as pr_apply_defense
from team_strength import apply_runs_delta as ts_apply_runs_delta

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)
MAX_PITCHERS = 10

# Extra innings start a runner on second (2020 onward, permanent for the
# regular season since 2023). Base code 2 is a runner on second.
GHOST_RUNNER_BASE = 2


def _load_sim_calibration():
    """End-to-end outcome-rate correction, if one has been fitted.

    Applied on top of the component-level normalisation correction because the
    simulator was still over-producing extra-base hits after it -- doubles
    +4.4%, triples +11.7%, home runs +3.8% on 2024, worth about 0.3 runs a
    game. Four candidate mechanisms were ruled out, so this measures the error
    where it shows up and corrects it there. See fit_sim_calibration.py; it is
    a patch and must be re-fitted whenever the model changes.
    """
    import json
    import os
    path = os.path.join(config.PROCESSED_DIR, "sim_calibration.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        return np.array(blob["calibration"], dtype=float)
    except Exception:      # noqa: BLE001 - a bad patch file must not break runs
        return None


SIM_CALIBRATION = _load_sim_calibration()


def _load_k_calibration():
    """Slope correction for strikeouts, if one has been fitted.

    Distinct from SIM_CALIBRATION, which is one multiplier per outcome and so
    can only move the level. The strikeout error is a *slope*: the model
    under-predicts low-K matchups and over-predicts high-K ones, because the
    odds ratio spreads strikeout probability too wide. On 2025 the top bucket
    of starts was over-projected by 0.47 K while the bottom was accurate.

    Aimed squarely at strikeout props, which live in that top bucket. See
    fit_k_calibration.py.
    """
    import json
    import os
    path = os.path.join(config.PROCESSED_DIR, "k_calibration.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        return float(blob["a"]), float(blob["b"])
    except Exception:
        return None


K_CALIBRATION = _load_k_calibration()

K_IDX = OUTCOMES.index("K")
BB_IDX = OUTCOMES.index("BB")
HBP_IDX = OUTCOMES.index("HBP")
IDX_1B = OUTCOMES.index("1B")
IDX_2B = OUTCOMES.index("2B")
IDX_3B = OUTCOMES.index("3B")
IDX_HR = OUTCOMES.index("HR")
HIT_IDX = np.array([IDX_1B, IDX_2B, IDX_3B, IDX_HR])


def build_runner_map(max_runs=4):
    """Canonical mapping of who ends up on which base after a plate appearance.

    The transition tables give the resulting base *occupancy* but not runner
    identity, and the simulator needs identity to carry each runner's speed
    forward. This resolves it with the ordinary rules of the game:

    * Runners advance in order and cannot pass one another.
    * The lead runners are the ones who score, so `runs` of them come off the
      front.
    * When more runners survive than there are bases to hold them, the excess
      came off the front too -- a force play retires the lead runner, and a
      double play starts with him.
    * If the batter reached, he takes the lowest occupied base and the
      surviving runners fill the higher ones in order.

    Returns an array indexed [base_before, base_after, runs, batter_reached]
    holding three sources, one per base (1st, 2nd, 3rd): -2 empty, -1 the
    batter, or 0/1/2 for the runner previously on 1st/2nd/3rd.
    """
    m = np.full((8, 8, max_runs + 1, 2, 3), -2, dtype=np.int8)
    for before_code in range(8):
        # Lead first: 3rd is bit 2, 2nd is bit 1, 1st is bit 0.
        before = [i for i in (2, 1, 0) if (before_code >> i) & 1]
        for after_code in range(8):
            after = [i for i in (2, 1, 0) if (after_code >> i) & 1]
            for runs in range(max_runs + 1):
                for reached in (0, 1):
                    survivors = before[runs:] if runs <= len(before) else []
                    slots = after[:-1] if (reached and after) else list(after)
                    if len(survivors) > len(slots):
                        survivors = survivors[len(survivors) - len(slots):]

                    row = np.full(3, -2, dtype=np.int8)
                    for k, base in enumerate(slots):
                        if k < len(survivors):
                            row[base] = survivors[k]
                    if reached and after:
                        row[after[-1]] = -1
                    m[before_code, after_code, runs, reached] = row
    return m


RUNNER_MAP = build_runner_map()


def build_scorer_map(max_runs=4):
    """Which occupied bases scored, given the base state and runs on the play.

    Lead runners score first -- the same convention `RUNNER_MAP` uses, so the
    two stay consistent. When runs exceed the number of runners aboard, the
    surplus is the batter himself, which is how a home run credits its hitter
    with a run without any special case.
    """
    m = np.zeros((8, max_runs + 1, 3), dtype=bool)
    for before in range(8):
        occupied = [i for i in (2, 1, 0) if (before >> i) & 1]
        for runs in range(max_runs + 1):
            for base in occupied[:min(runs, len(occupied))]:
                m[before, runs, base] = True
    return m


SCORER_MAP = build_scorer_map()

# Fantasy scoring: single 3, double 5, triple 8, home run 10, run 2, RBI 2,
# walk 2. Applied to the per-batter counts the simulator accumulates.
FANTASY_POINTS = {"1B": 3.0, "2B": 5.0, "3B": 8.0, "HR": 10.0,
                  "R": 2.0, "RBI": 2.0, "BB": 2.0}

# Whether the batter ends up on a base, by outcome. A home run scores him
# rather than leaving him on base, and on an out he is assumed retired -- the
# fielder's choice case, where he reaches while a runner is retired, keeps the
# occupancy right and only misplaces one speed value.
BATTER_REACHES = np.array(
    [0 if o in ("K", "OUT", "HR") else 1 for o in OUTCOMES], dtype=np.int8)


def speed_bucket(values, edges, n=3):
    """Bucket speeds with the same thresholds the transition table was built on."""
    if edges is None or len(edges) == 0:
        return np.full(np.shape(values), n // 2, dtype=np.int8)
    return np.clip(np.searchsorted(edges, values), 0, n - 1).astype(np.int8)


@dataclass
class HookRule:
    """Fixed usage baseline. Pitchers change only at half-inning boundaries.

    Kept for comparison. It pulls every starter at 24 batters faced, against a
    2024-2026 reality of 21.87 with a standard deviation of 4.65 -- too late,
    and with no variance at all in a league where 30.7% of starts do not reach
    the fifth inning.
    """
    starter_bf: int = 24
    reliever_bf: int = 4

    def should_pull(self, pitcher_slot, bf):
        limit = np.where(pitcher_slot == 0, self.starter_bf, self.reliever_bf)
        return bf >= limit


# Average pitches thrown in a plate appearance ending in each outcome,
# MEASURED from 2023+ rather than assumed. The original hand-picked values ran
# 3.0% high in aggregate -- 4.015 per plate appearance against an actual 3.897,
# which is +2.6 pitches over a 22-batter start. Since pitch count is the
# strongest driver of the removal hazard, that inflation pulled starters early
# all by itself. The errors were not uniform either: walks were understated by
# 0.44 pitches while every ball-in-play outcome was overstated by 0.2-0.4.
PITCHES_PER_OUTCOME = np.array([4.87, 5.74, 3.09, 3.36, 3.36, 3.33, 3.42, 3.31])


class HazardHook:
    """Stochastic hook driven by the fitted removal hazard.

    Evaluated after every batter the starter faces, so the decision responds to
    workload, damage, inning, and base-out state rather than firing at a fixed
    count. Relievers still turn over on a simple rule -- modelling reliever
    removal is a separate problem and not what dominates the innings split.
    """

    def __init__(self, hook_model, feature_names, reliever_bf=3,
                 reliever_exit_prob=0.50, hazard_scale=1.0):
        self.m = hook_model
        self.names = list(feature_names)
        # The fitted hazard pulls starters about half an out early -- 15.12
        # projected outs against 15.66 actual on the 2025 backtest, uniform
        # across workload bands rather than concentrated anywhere. That
        # shortfall costs roughly 0.17 strikeouts a start, which was masking
        # the opposite-signed strikeout rate bias: the two nearly cancelled in
        # the average while both were wrong. Scaling the hazard below 1
        # lengthens starts; see fit_hook_scale.py.
        self.hazard_scale = hazard_scale
        # Three-batter minimum, then an exit probability at each inning
        # boundary. Real relief appearances average 4.83 batters faced with a
        # spike at exactly three (27.8% of outings), which is the rule showing
        # up directly in the data.
        self.reliever_bf = reliever_bf
        self.reliever_exit_prob = reliever_exit_prob

    def starter_hazard(self, feats):
        X = np.column_stack([feats[n] for n in self.names])
        h = self.m.hazard(X)
        if self.hazard_scale != 1.0:
            h = np.clip(h * self.hazard_scale, 0.0, 1.0)
        return h


def simulate(cum_probs, table, n_sims, rng, hook=None, max_steps=400,
             tto_mult=None, lineup_speed=None, ghost_speed=None,
             starter_rest=None, starter_expected_outs=None,
             starter_is_opener=None, diagnostics=False,
             dispersion_sigma=0.0, pen_hand=None, lineup_side=None,
             match_prob=0.5, initial_state=None, hook_sigma=0.0,
             pinch=None):
    """Simulate `n_sims` complete games.

    `cum_probs` has shape (2, MAX_PITCHERS, 9, 2, N_OUT): the first axis is which
    team is batting (0 = away, 1 = home), the second is the fielding team's
    current pitcher slot, the third the batting order slot, and the fourth the
    substitution state (0 = the slot's starter, 1 = a pinch hitter has entered).
    Values are cumulative outcome probabilities.

    `pinch` is `(weights, leverage_table)` from `pinch_hitting.load_hazard()`
    and `leverage.load_leverage_table()`. Left as None the lineup never changes,
    which is how this ran before substitutions existed.

    `tto_mult` is an optional (4, N_OUT) array of times-through-order
    multipliers applied to the starter only, indexed by turn through the order.

    Returns (away_runs, home_runs, extras) arrays of length n_sims.
    """
    hook = hook or HookRule()
    n = n_sims

    r_edges, b_edges = getattr(table, "speed_edges", (np.array([]), np.array([])))
    use_speed = lineup_speed is not None and len(r_edges) > 0
    if use_speed:
        # Speed of whoever occupies each base, carried through the game so a
        # fast runner on second actually scores from second more often.
        base_speed = np.zeros((n, 3), dtype=np.float32)
        bat_bucket_tbl = speed_bucket(lineup_speed, b_edges)   # (2, 9)
        ghost = float(ghost_speed if ghost_speed is not None
                      else np.mean(lineup_speed))

    inning = np.ones(n, dtype=np.int16)
    is_top = np.ones(n, dtype=bool)
    outs = np.zeros(n, dtype=np.int8)
    base = np.zeros(n, dtype=np.int8)
    score = np.zeros((n, 2), dtype=np.int16)      # [:, 0] away, [:, 1] home
    bat_idx = np.zeros((n, 2), dtype=np.int8)
    pit_idx = np.zeros((n, 2), dtype=np.int8)     # fielding team's pitcher slot

    # Substitution state per lineup slot. Once a pinch hitter enters he stays,
    # which is both what happens and what makes the bench a depleting resource.
    subbed = np.zeros((n, 2, 9), dtype=bool)
    ph_w = ph_li = None
    if pinch is not None:
        ph_w, ph_li = pinch
    bf = np.zeros((n, 2), dtype=np.int16)         # batters faced by that pitcher
    tto = np.zeros((n, 2), dtype=np.int16)        # batters faced by the starter
    sp_pitches = np.zeros((n, 2), dtype=np.float32)   # starter pitch count
    sp_runs = np.zeros((n, 2), dtype=np.float32)      # runs charged to the starter
    sp_outs = np.zeros((n, 2), dtype=np.float32)      # outs the starter recorded

    sp_k = np.zeros((n, 2), dtype=np.int16)           # starter strikeouts
    sp_hits = np.zeros((n, 2), dtype=np.int16)        # hits allowed by the starter
    sp_bb = np.zeros((n, 2), dtype=np.int16)          # walks issued by the starter
    # Hit batsmen tracked apart from walks: Yahoo scores them alike (-0.9)
    # but `starter_bb` is already published as walks in the prop output,
    # and folding HBP into it would quietly change that meaning.
    sp_hbp = np.zeros((n, 2), dtype=np.int16)

    # Per-batter counting stats, indexed [sim, team, lineup slot]. Tracking
    # which lineup slot occupies each base is what makes runs scored
    # attributable: the transition tables say a run crossed, not who crossed.
    track = diagnostics
    outcome_counts = np.zeros(N_OUT, dtype=np.int64)
    # Plate appearances taken by a substitute, against the total, so the
    # simulated substitution rate can be checked against the real one (3.74% of
    # PAs on 2025) rather than assumed correct.
    sub_pa_count = np.zeros(2, dtype=np.int64)
    if track:
        bat_1b = np.zeros((n, 2, 9), dtype=np.int16)
        bat_2b = np.zeros((n, 2, 9), dtype=np.int16)
        bat_3b = np.zeros((n, 2, 9), dtype=np.int16)
        bat_hr = np.zeros((n, 2, 9), dtype=np.int16)
        bat_bb = np.zeros((n, 2, 9), dtype=np.int16)
        bat_r = np.zeros((n, 2, 9), dtype=np.int16)
        bat_rbi = np.zeros((n, 2, 9), dtype=np.int16)
        base_slot = np.full((n, 3), -1, dtype=np.int8)
        base_sub = np.zeros((n, 3), dtype=bool)
        if initial_state and "base_slots" in initial_state:
            base_slot[:] = np.asarray(initial_state["base_slots"], dtype=np.int8)
    done = np.zeros(n, dtype=bool)

    # Start from an arbitrary game state rather than the top of the first.
    # This is what lets a decision be evaluated: freeze the situation, branch
    # on the choice, and run each branch forward to a win probability.
    if initial_state:
        s = initial_state
        if "inning" in s:
            inning[:] = s["inning"]
        if "is_top" in s:
            is_top[:] = s["is_top"]
        if "outs" in s:
            outs[:] = s["outs"]
        if "base_code" in s:
            base[:] = s["base_code"]
        if "score" in s:
            score[:] = np.asarray(s["score"], dtype=np.int16)
        if "bat_idx" in s:
            bat_idx[:] = np.asarray(s["bat_idx"], dtype=np.int8)
        if "pit_idx" in s:
            pit_idx[:] = np.asarray(s["pit_idx"], dtype=np.int8)
        if "bf" in s:
            bf[:] = np.asarray(s["bf"], dtype=np.int16)
        if "starter_bf" in s:
            tto[:] = np.asarray(s["starter_bf"], dtype=np.int16)
        if "starter_pitches" in s:
            sp_pitches[:] = np.asarray(s["starter_pitches"], dtype=np.float32)
        if "starter_outs" in s:
            sp_outs[:] = np.asarray(s["starter_outs"], dtype=np.float32)
        if "starter_runs" in s:
            sp_runs[:] = np.asarray(s["starter_runs"], dtype=np.float32)
        # Who is on base matters, not just that the bases are occupied: a fast
        # runner on second scores from second far more often.
        if use_speed and "base_speed" in s:
            base_speed[:] = np.asarray(s["base_speed"], dtype=np.float32)

    use_hazard = isinstance(hook, HazardHook)
    rest = np.asarray(starter_rest if starter_rest is not None else [5.0, 5.0],
                      dtype=float)
    exp_outs = np.asarray(
        starter_expected_outs if starter_expected_outs is not None
        else [15.5, 15.5], dtype=float)
    # An opener is a reliever who happens to throw the first inning. Flagging
    # him lets the hazard pull him after roughly an inning, which in turn hands
    # the ball to whoever the bullpen ordering puts first -- normally the bulk
    # pitcher, since he is the one who typically enters in the second.
    opener = np.asarray(
        starter_is_opener if starter_is_opener is not None else [0.0, 0.0],
        dtype=float)

    # Game-level dispersion, drawn independently per team.
    #
    # Independently, not shared: the covariance between the two teams' run
    # totals in a real game is -0.011 (correlation -0.0011), essentially zero.
    # A shared shock -- weather, umpire zone, a hot day at altitude -- would
    # create positive covariance, and the data says there is none to create.
    # What is missing is team-specific: an offense that collectively squares
    # everything up, or a staff that has nothing on a given night.
    # Per-start hook shock: a persistent shift in the starter's removal hazard,
    # held for a whole game. MEASURED AND REJECTED -- leave at 0.
    #
    # The reasoning was that independent per-plate-appearance draws cannot
    # produce the correlation in a real start, where a pitcher who is cruising
    # carries a low hazard all night. The reasoning was sound and the premise
    # was false: simulated start lengths are already 11% WIDER than actual
    # (sd 4.42 against 3.97), with more seven-inning starts than reality
    # (12.1% against 10.3%). Every non-zero sigma makes the fit worse.
    #
    # The apparent compression that motivated this was an analysis error --
    # binning starts by their *mean* simulated outs, which of course rarely
    # exceeds 18 for a distribution centred near 15. That is the spread of
    # per-start averages, not of outcomes.
    #
    # Retained at zero because the parameter is harmless and the comment is
    # cheaper than someone rediscovering this.
    if hook_sigma > 0:
        hook_shock = rng.normal(0.0, hook_sigma, size=(n, 2))
    else:
        hook_shock = None

    if dispersion_sigma > 0:
        # Centred so the multiplier averages exactly 1. A raw lognormal has
        # mean exp(sigma^2 / 2), which at sigma 0.18 lifts on-base rates by
        # 1.6% and, because run scoring is convex in those rates, team runs by
        # over 4% -- the spread would be right and the mean wrong.
        shock = np.exp(rng.normal(-0.5 * dispersion_sigma ** 2,
                                  dispersion_sigma, size=(n, 2)))
    else:
        shock = None
    # Strikeouts and outs are left alone; the shock scales the ways a batter
    # reaches, and renormalization takes the mass out of the outs.
    reach_idx = np.array([OUTCOMES.index(o) for o in
                          ("BB", "HBP", "1B", "2B", "3B", "HR")])

    if pen_hand is not None and lineup_side is None:
        pen_hand = None   # cannot match handedness without knowing who is due

    rows = np.arange(n)

    for _ in range(max_steps):
        if done.all():
            break
        live = ~done

        # All randomness for this step is drawn up front, unconditionally.
        # Common random numbers require every branch of a decision to consume
        # the stream identically; a draw made only when some condition holds
        # desynchronises the branches the moment they diverge, which destroys
        # the correlation the whole technique depends on.
        u_outcome = rng.random(n)
        u_transition = rng.random(n)
        u_hazard = rng.random(n)
        u_relief = rng.random(n)
        u_match = rng.random(n)

        bat_team = np.where(is_top, 0, 1)          # who is hitting
        fld_team = 1 - bat_team                    # who is pitching

        b = bat_idx[rows, bat_team]
        p = pit_idx[rows, fld_team]

        # Pinch-hit decision, evaluated before the plate appearance resolves.
        # Drawn unconditionally so the random stream stays aligned for common
        # random numbers, exactly as the hook draw is.
        u_pinch = rng.random(n)
        if ph_w is not None:
            already = subbed[rows, bat_team, b]
            margin = score[rows, bat_team] - score[rows, fld_team]
            late = (inning >= 7).astype(np.float64)
            very_late = (inning >= 9).astype(np.float64)
            trailing_close = ((margin < 0) & (margin >= -3)).astype(np.float64)
            reliever_in = (p > 0).astype(np.float64)
            if lineup_side is not None and pen_hand is not None:
                disadv = (lineup_side[bat_team, b]
                          == pen_hand[fld_team, p]).astype(np.float64)
            else:
                disadv = np.zeros(n)
            if ph_li is not None:
                import leverage as _lev
                st = _lev.state_index(inning, is_top, outs, base, margin)
                li = ph_li[np.clip(st, 0, len(ph_li) - 1)]
            else:
                li = np.ones(n)
            z = (ph_w[0] + ph_w[1] * late + ph_w[2] * very_late
                 + ph_w[3] * np.log(np.clip(li, 0.05, 20.0))
                 + ph_w[4] * disadv + ph_w[5] * trailing_close
                 + ph_w[6] * reliever_in)
            fires = (u_pinch < 1.0 / (1.0 + np.exp(-z))) & ~already
            if fires.any():
                subbed[rows[fires], bat_team[fires], b[fires]] = True

        sub_ax = subbed[rows, bat_team, b].astype(np.int8)
        probs = cum_probs[bat_team, p, b, sub_ax]   # (n, N_OUT) cumulative

        # Times through the order applies to the starter only, and the
        # dispersion shock to whichever team is batting. Both act on densities,
        # so the cumulative vector is unpacked once and re-accumulated once.
        if tto_mult is not None or shock is not None:
            dens = np.diff(probs, axis=1, prepend=0.0)

            if tto_mult is not None:
                starter = p == 0
                turn = np.clip(tto[rows, fld_team] // 9, 0, 3)
                dens = dens * np.where(starter[:, None], tto_mult[turn], 1.0)

            if shock is not None:
                dens[:, reach_idx] *= shock[rows, bat_team][:, None]

            probs = np.cumsum(dens, axis=1)
            probs = probs / probs[:, -1:]

        outcome = (u_outcome[:, None] >= probs).sum(axis=1)
        np.clip(outcome, 0, N_OUT - 1, out=outcome)

        cell = (outcome.astype(np.int32) * 8 + base) * 3 + outs
        if use_speed:
            # Runner bucket is the mean speed across occupied bases; with the
            # bases empty there is no runner to condition on, so it falls to
            # the middle bucket.
            occ = np.stack([(base >> k) & 1 for k in range(3)], axis=1)
            n_occ = occ.sum(axis=1)
            mean_speed = np.where(
                n_occ > 0,
                (base_speed * occ).sum(axis=1) / np.maximum(n_occ, 1), 0.0)
            rb = np.where(n_occ > 0, speed_bucket(mean_speed, r_edges), 1)
            bb = bat_bucket_tbl[bat_team, b]
            cell = (cell * 3 + rb) * 3 + bb

        base_before = base.copy()
        runs, outs_added, base_after = table.sample(cell, u_transition)

        # Apply only to live simulations.
        score[rows, bat_team] += np.where(live, runs, 0).astype(np.int16)
        new_outs = outs + np.where(live, outs_added, 0).astype(np.int8)
        half_over = live & (new_outs >= 3)

        bat_idx[rows, bat_team] = np.where(
            live, (bat_idx[rows, bat_team] + 1) % 9, bat_idx[rows, bat_team])
        bf[rows, fld_team] += live.astype(np.int16)
        is_sp = p == 0
        tto[rows, fld_team] += (live & is_sp).astype(np.int16)
        sp_pitches[rows, fld_team] += np.where(
            live & is_sp, PITCHES_PER_OUTCOME[outcome], 0.0).astype(np.float32)
        sp_runs[rows, fld_team] += np.where(live & is_sp, runs, 0).astype(np.float32)
        sp_outs[rows, fld_team] += np.where(
            live & is_sp, outs_added, 0).astype(np.float32)

        # Per-pitcher counting stats, which is what player props are priced on.
        # These come from the same draws as everything else, so the strikeout
        # total and the innings total stay coupled -- a start that ends early
        # cannot also rack up strikeouts, which is exactly the dependence a
        # rate-times-expected-batters estimate throws away.
        active = live & is_sp
        sp_k[rows, fld_team] += (active & (outcome == K_IDX)).astype(np.int16)
        sp_bb[rows, fld_team] += (active & (outcome == BB_IDX)).astype(np.int16)
        sp_hbp[rows, fld_team] += (active & (outcome == HBP_IDX)).astype(np.int16)
        sp_hits[rows, fld_team] += (active & np.isin(outcome, HIT_IDX)).astype(np.int16)

        if track:
            # Every outcome the simulator actually produces, pooled. Comparing
            # these against the season's real rates splits a run-level bias in
            # two: if the rates match but the runs do not, the state machine is
            # converting outcomes to runs too generously; if the rates are off,
            # the projection or context layer is.
            np.add.at(outcome_counts, outcome[live], 1)
            sub_pa_count[0] += int(sub_ax[live].sum())
            sub_pa_count[1] += int(live.sum())

            # Credit only while the slot still holds its starter. The
            # counters are indexed by lineup slot, not player id, and props
            # maps a slot back to the man who started there -- so without this
            # gate a pinch hitter's production lands on the starter's line and
            # the starter keeps accruing plate appearances he never took. That
            # left hitter counting props over-projected by 2.5-3.6%.
            own = live & (sub_ax == 0)
            for idx, arr in ((IDX_1B, bat_1b), (IDX_2B, bat_2b),
                             (IDX_3B, bat_3b), (IDX_HR, bat_hr),
                             (BB_IDX, bat_bb)):
                arr[rows, bat_team, b] += (own & (outcome == idx)).astype(np.int16)

            # Runs on the play are credited to the batter as runs batted in.
            # This slightly overcounts -- a run that scores on an error or a
            # double play is not an RBI -- but those are a small share.
            bat_rbi[rows, bat_team, b] += np.where(own, runs, 0).astype(np.int16)

            # Who actually crossed the plate.
            scorers = SCORER_MAP[base_before, np.clip(runs, 0, 4)]   # (n, 3)
            for base in range(3):
                sc = (live & scorers[:, base] & (base_slot[:, base] >= 0)
                      & ~base_sub[:, base])
                if sc.any():
                    np.add.at(bat_r, (rows[sc], bat_team[sc],
                                      base_slot[sc, base]), 1)
            # Runs beyond the runners aboard are the batter himself.
            n_on = ((base_before & 1) + ((base_before >> 1) & 1)
                    + ((base_before >> 2) & 1))
            self_score = own & (runs > n_on)
            if self_score.any():
                np.add.at(bat_r, (rows[self_score], bat_team[self_score],
                                  b[self_score]), 1)

        outs = np.where(half_over, 0, new_outs).astype(np.int8)
        base = np.where(half_over, 0, base_after).astype(np.int8)

        if use_speed or track:
            reached = BATTER_REACHES[outcome]
            src = RUNNER_MAP[base_before, base_after.astype(np.int8),
                             np.clip(runs, 0, 4), reached]        # (n, 3)
            take = np.maximum(src, 0).astype(np.intp)

        if use_speed:
            carried = np.take_along_axis(base_speed, take, axis=1)
            batter_speed = lineup_speed[bat_team, b][:, None]
            new_speed = np.where(src == -1, batter_speed,
                                 np.where(src >= 0, carried, 0.0))
            base_speed = np.where(live[:, None], new_speed,
                                  base_speed).astype(np.float32)
            base_speed[half_over] = 0.0

        if track:
            # Runner identity follows the same canonical mapping as speed, so
            # a run credited to a base is credited to the right hitter.
            carried_slot = np.take_along_axis(base_slot, take, axis=1)
            new_slot = np.where(src == -1, b[:, None],
                                np.where(src >= 0, carried_slot, -1))

            # Whether each base's occupant reached as a substitute, carried
            # by the same mapping as speed and slot. `base_slot` records only
            # which lineup *slot* is on the bag, so when a pinch hitter reaches
            # and later scores, the run would otherwise be credited to the man
            # who started in that slot. Bases clear at every half-inning end,
            # so this only ever has to be right within an inning.
            carried_sub = np.take_along_axis(base_sub, take, axis=1)
            new_sub = np.where(src == -1, (sub_ax == 1)[:, None],
                               np.where(src >= 0, carried_sub, False))
            base_sub = np.where(live[:, None], new_sub, base_sub)
            base_sub[half_over] = False

            base_slot = np.where(live[:, None], new_slot,
                                 base_slot).astype(np.int8)
            base_slot[half_over] = -1

        # ---- hook decision, evaluated after every batter the starter faces
        # rather than at a fixed count. Features describe the state the manager
        # is looking at, which is why the post-plate-appearance base-out state
        # is captured here before the half-inning rollover clears it.
        if use_hazard:
            runners_after = ((base_after & 1) + ((base_after >> 1) & 1)
                             + ((base_after >> 2) & 1))
            runners_now = np.where(half_over, 0, runners_after).astype(float)
            sp_bf = bf[rows, fld_team].astype(float)
            sp_o = sp_outs[rows, fld_team].astype(float)
            sp_r = sp_runs[rows, fld_team].astype(float)
            denom = np.maximum(sp_o, 3.0)
            feats = {
                "outs_recorded": sp_o,
                "pitches": sp_pitches[rows, fld_team].astype(float),
                "tto_next": np.clip(np.floor(sp_bf / 9.0), 0, 3) + 1.0,
                "runs_allowed": sp_r,
                "runs_per_out": sp_r / denom,
                "traffic": np.maximum(sp_bf - sp_o, 0.0) / denom,
                "score_diff": (score[rows, fld_team]
                               - score[rows, bat_team]).astype(float),
                "leading": (score[rows, fld_team]
                            > score[rows, bat_team]).astype(float),
                "inning": inning.astype(float),
                "outs_after": np.where(half_over, 3, new_outs).astype(float),
                "inning_ended": half_over.astype(float),
                "runners_on": runners_now,
                "bases_empty": (runners_now == 0).astype(float),
                "days_rest": rest[fld_team],
                "expected_outs": exp_outs[fld_team],
                "is_opener": opener[fld_team],
            }
            pc = sp_pitches[rows, fld_team].astype(float)
            for thresh in (60, 80, 95):
                feats[f"pitches_over_{thresh}"] = np.maximum(pc - thresh, 0.0)

            # Same centring as the training frame, or the interactions mean
            # something different at fit time than at simulation time.
            wh = (exp_outs[fld_team] - 15.5) / 5.0
            pcn = (pc - 80.0) / 20.0
            dmg = feats["runs_per_out"] - 0.115
            feats["workhorse_x_pitches"] = wh * pcn
            feats["workhorse_x_damage"] = wh * dmg
            feats["pitches_x_damage"] = pcn * dmg
            feats["outs_12_15"] = ((sp_o >= 12) & (sp_o < 15)).astype(float)
            feats["outs_15_18"] = ((sp_o >= 15) & (sp_o < 18)).astype(float)
            feats["outs_18_21"] = ((sp_o >= 18) & (sp_o < 21)).astype(float)
            feats["outs_21_plus"] = (sp_o >= 21).astype(float)
            hz = hook.starter_hazard(feats)
            if hook_shock is not None:
                # Shift in log-odds so the shock scales the hazard
                # multiplicatively at low rates -- which is the regime that
                # matters, since a per-batter removal hazard is a few percent
                # for most of a start.
                z = np.log(np.clip(hz, 1e-9, 1 - 1e-9)
                           / np.clip(1 - hz, 1e-9, 1.0))
                hz = 1.0 / (1.0 + np.exp(-(z + hook_shock[rows, fld_team])))
            pull_sp = live & is_sp & (u_hazard < hz)
            pit_idx[rows, fld_team] = np.where(pull_sp, 1, pit_idx[rows, fld_team])
            bf[rows, fld_team] = np.where(pull_sp, 0, bf[rows, fld_team])

        # ---- walk-off: the home team taking the lead in the ninth or later
        # ends the game immediately, whether or not the play recorded an out.
        # Checked before the rollover below, so `roll` correctly skips these.
        done |= (live & ~is_top & (inning >= 9) & (score[:, 1] > score[:, 0]))

        # ---- half-inning rollover
        roll = half_over & ~done
        if roll.any():
            was_top = is_top & roll

            # Reliever turnover at half-inning boundaries. With the hazard hook
            # active the starter has already been handled per plate appearance,
            # so only relievers are considered here.
            if use_hazard:
                # Relievers turn over at inning boundaries, but not every time:
                # a flat rule produced 4.80 pitchers per team-game against an
                # actual 4.28, because it never lets anyone go two innings.
                # The three-batter minimum puts a hard floor under it.
                eligible = (roll & (pit_idx[rows, fld_team] > 0)
                            & (bf[rows, fld_team] >= hook.reliever_bf))
                pull = eligible & (u_relief < hook.reliever_exit_prob)
            else:
                pull = roll & hook.should_pull(pit_idx[rows, fld_team],
                                               bf[rows, fld_team])
            if pull.any():
                nxt = np.minimum(pit_idx[rows, fld_team] + 1, MAX_PITCHERS - 1)

                # Handedness preference. Managers get a same-handed matchup
                # 58.3% of the time against a 52.3% chance baseline, so the
                # effect is real but mild -- the three-batter minimum limits
                # how much matching is possible. Applied stochastically to
                # avoid overstating it.
                if pen_hand is not None:
                    due_side = lineup_side[bat_team, bat_idx[rows, bat_team]]
                    cand = np.minimum(nxt + 1, MAX_PITCHERS - 1)
                    cand_matches = pen_hand[fld_team, cand] == due_side
                    nxt_matches = pen_hand[fld_team, nxt] == due_side
                    swap = (cand_matches & ~nxt_matches & (u_match < match_prob))
                    nxt = np.where(swap, cand, nxt)

                pit_idx[rows, fld_team] = np.where(pull, nxt,
                                                   pit_idx[rows, fld_team])
                bf[rows, fld_team] = np.where(pull, 0, bf[rows, fld_team])

            inning = np.where(roll & ~was_top, inning + 1, inning).astype(np.int16)
            is_top = np.where(roll, ~is_top, is_top)

            # Ghost runner from the tenth.
            ghost_on = roll & (inning >= 10)
            base = np.where(ghost_on, GHOST_RUNNER_BASE,
                            np.where(roll, 0, base)).astype(np.int8)
            if use_speed and ghost_on.any():
                # The placed runner is the man who made the last out, so a
                # lineup-average speed is the right stand-in.
                base_speed[ghost_on, 1] = ghost

            # Game over after a completed ninth or later when someone leads.
            # The two thresholds differ because `inning` was just incremented
            # for bottom-half rollovers but not top-half ones: after the top of
            # the ninth `inning` is still 9, while after the bottom of the
            # ninth it already reads 10.
            # After the top half: the home team does not bat if it is ahead.
            end_after_top = roll & was_top & (inning >= 9) & (score[:, 1] > score[:, 0])
            # After the bottom half: any lead ends it.
            end_after_bot = (roll & ~was_top & (inning >= 10)
                             & (score[:, 0] != score[:, 1]))
            done |= end_after_top | end_after_bot

    extras = inning > 9
    if diagnostics:
        # `tto` counts batters the starter faced, which is exactly the workload
        # the hook model is supposed to reproduce.
        total_bases = bat_1b + 2 * bat_2b + 3 * bat_3b + 4 * bat_hr
        hits = bat_1b + bat_2b + bat_3b + bat_hr
        fantasy = (FANTASY_POINTS["1B"] * bat_1b
                   + FANTASY_POINTS["2B"] * bat_2b
                   + FANTASY_POINTS["3B"] * bat_3b
                   + FANTASY_POINTS["HR"] * bat_hr
                   + FANTASY_POINTS["R"] * bat_r
                   + FANTASY_POINTS["RBI"] * bat_rbi
                   + FANTASY_POINTS["BB"] * bat_bb)
        diag = {"starter_bf": tto.copy(), "starter_pitches": sp_pitches.copy(),
                "n_pitchers": pit_idx.copy() + 1,
                "outcome_counts": outcome_counts,
                "sub_pa": sub_pa_count,
                "starter_k": sp_k.copy(), "starter_outs": sp_outs.copy(),
                "starter_hits": sp_hits.copy(), "starter_bb": sp_bb.copy(),
                "starter_hbp": sp_hbp.copy(),
                "starter_runs": sp_runs.copy(),
                # Per-batter, indexed [sim, team, lineup slot].
                "bat_1b": bat_1b, "bat_2b": bat_2b, "bat_3b": bat_3b,
                "bat_hr": bat_hr, "bat_bb": bat_bb, "bat_r": bat_r,
                "bat_rbi": bat_rbi, "bat_hits": hits,
                "bat_total_bases": total_bases,
                "bat_hrr": hits + bat_r + bat_rbi,
                "bat_fantasy": fantasy}
        return score[:, 0].copy(), score[:, 1].copy(), extras, diag
    return score[:, 0].copy(), score[:, 1].copy(), extras


def league_reliever_vectors(pa, seasons=None):
    """League-average relief pitcher outcome rates, by batter side.

    Relievers are meaningfully better than starters, so filling the late
    innings with a generic *pitcher* rather than a generic *reliever* inflates
    simulated scoring.
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES) & ~pa["is_starter"]]
    if seasons:
        d = d[d["season"].isin(seasons)]

    out = {}
    for side in ["L", "R"]:
        s = d[d["stand"] == side]
        v = np.array([(s["outcome"] == o).sum() for o in OUTCOMES], dtype=float)
        out[side] = v / v.sum()
    return out


def _recalibrate_k(v):
    """logit(p_K') = a + b*logit(p_K), with the rest of the vector rescaled.

    Returns a distribution: the seven non-strikeout outcomes are scaled by a
    common factor to fill 1 - p_K', which preserves their proportions among
    themselves. Only the strikeout share moves.
    """
    a, b = K_CALIBRATION
    p = float(np.clip(v[K_IDX], 1e-9, 1 - 1e-9))
    q = 1.0 / (1.0 + np.exp(-(a + b * np.log(p / (1.0 - p)))))
    rest = v.sum() - v[K_IDX]
    if rest <= 0:
        return v
    out = v * ((1.0 - q) / rest)
    out[K_IDX] = q
    return out


def build_cum_probs(model, away_lineup, home_lineup, away_stands, home_stands,
                    away_starter, home_starter, away_hand, home_hand,
                    reliever_vec, park_mult=None, calibration=None,
                    home_advantage=None, defense_h=None, bullpen_arms=None,
                    team_runs_delta=None, linear_weights=None,
                    pinch_vec=None):
    """Assemble the (2, MAX_PITCHERS, 9, N_OUT) cumulative probability array.

    Slot 0 of each fielding team is its actual starter; every later slot is the
    generic reliever, which is what makes this the fixed-usage baseline.
    """
    # Trailing axis of length 2 is the substitution state: 0 while the slot
    # holds its starter, 1 once a pinch hitter has entered. Both are built
    # up front so the simulation loop only ever indexes, never rebuilds.
    cum = np.zeros((2, MAX_PITCHERS, 9, 2, N_OUT))
    lg = model.league

    hfa = home_advantage or {}
    specs = [
        # (team index, lineup, stands, opposing starter, his hand, HFA vector)
        (0, away_lineup, away_stands, home_starter, home_hand,
         hfa.get("away")),
        (1, home_lineup, home_stands, away_starter, away_hand,
         hfa.get("home")),
    ]

    for team, lineup, stands, starter_id, opp_hand, hfa_mult in specs:
        for slot in range(9):
            bid, side = lineup[slot], stands[slot]
            b = model.batter_vector(bid, side, opp_hand)

            # Slot 0 is the starter. Slots 1+ are the team's actual relief arms
            # in the order a manager would use them, falling back to a generic
            # league-average reliever when no bullpen is supplied.
            pen = None if bullpen_arms is None else bullpen_arms[1 - team]
            if pen is not None and len(pen[0]):
                # Each reliever gets the same platoon treatment as the starter,
                # so a left-handed arm actually plays as one against this hitter.
                arms = [(model.pitcher_vector(int(pid), hand, side),
                         model.denominator(side, hand, use_platoon=True))
                        for pid, hand in zip(pen[0], pen[1])]
            else:
                # Applied hand-agnostically: with no identity there is no
                # platoon split to resolve, and matching the denominator keeps
                # the odds ratio consistent.
                arms = [(reliever_vec[side], lg)]

            for pslot, (p, denom) in enumerate([
                (model.pitcher_vector(starter_id, opp_hand, side),
                 model.denominator(side, opp_hand, use_platoon=True)),
            ] + arms):
                if pslot >= MAX_PITCHERS:
                    break
                v = b * p / np.maximum(denom, 1e-12)
                if park_mult is not None:
                    v = v * park_mult
                if hfa_mult is not None:
                    v = v * hfa_mult
                if calibration is not None:
                    v = v * calibration
                if SIM_CALIBRATION is not None:
                    v = v * SIM_CALIBRATION
                v = v / v.sum()

                # Strikeout slope correction, applied after normalisation
                # because it is a recalibration of a probability rather than a
                # scaling of a rate. The non-strikeout outcomes keep their
                # relative proportions and absorb the difference.
                if K_CALIBRATION is not None:
                    v = _recalibrate_k(v)

                # Team defense converts balls in play into outs. Applied after
                # normalization because it moves probability between outcomes
                # rather than scaling them, and only to fieldable balls -- no
                # fielder catches a home run, and range does not act on
                # strikeouts, walks, or hit batsmen.
                if defense_h is not None:
                    h = defense_h[1 - team]   # the defense is the other team
                    if h:
                        v = pr_apply_defense(v, h)

                # Team-strength correction from Elo. The batting team's share
                # lifts (or lowers) its offence; the fielding team's share acts
                # on its run prevention, which from this batter's perspective
                # is the same lever with the opposite sign.
                if team_runs_delta is not None and linear_weights is not None:
                    net = team_runs_delta[team] - team_runs_delta[1 - team]
                    if net:
                        v = ts_apply_runs_delta(v, net, linear_weights)

                cum[team, pslot, slot, 0] = np.cumsum(v)

                # Index 1 of the substitution axis is the same slot after a
                # pinch hitter has entered. The bench bat is modelled as the
                # league-average pinch hitter with the platoon advantage --
                # which is how they are actually chosen -- rather than as a
                # named player, because individual bench identities would need
                # per-game bench rosters and would add more variance than the
                # effect being captured.
                if pinch_vec is not None:
                    ph_side = "L" if opp_hand == "R" else "R"
                    pb = pinch_vec.get(ph_side)
                    if pb is None:
                        cum[team, pslot, slot, 1] = cum[team, pslot, slot, 0]
                    else:
                        pv = pb * p / np.maximum(
                            model.denominator(ph_side, opp_hand,
                                              use_platoon=True), 1e-12)
                        if park_mult is not None:
                            pv = pv * park_mult
                        if hfa_mult is not None:
                            pv = pv * hfa_mult
                        if calibration is not None:
                            pv = pv * calibration
                        if SIM_CALIBRATION is not None:
                            pv = pv * SIM_CALIBRATION
                        pv = pv / pv.sum()
                        if K_CALIBRATION is not None:
                            pv = _recalibrate_k(pv)
                        if defense_h is not None:
                            hh = defense_h[1 - team]
                            if hh:
                                pv = pr_apply_defense(pv, hh)
                        cum[team, pslot, slot, 1] = np.cumsum(pv)
                else:
                    cum[team, pslot, slot, 1] = cum[team, pslot, slot, 0]

            # Any pitcher slot beyond the arms available repeats the last one,
            # so a game that burns through the whole pen still resolves.
            n_used = min(1 + len(arms), MAX_PITCHERS)
            cum[team, n_used:, slot] = cum[team, n_used - 1, slot]

    cum[..., -1] = 1.0
    return cum
