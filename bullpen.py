"""Bullpen modelling: when the starter comes out, and who replaces him.

Three pieces, all of which the fixed-usage baseline got wrong:

**Hook model.** A discrete-time hazard evaluated after every batter the starter
faces: given workload, times through the order, damage allowed, score, inning,
and base-out state, what is the probability he does not face the next man?
The baseline pulled every starter at exactly 24 batters faced. Reality over
2024-2026 is a mean of 21.87 with a standard deviation of 4.65 -- so the
baseline ran starters roughly two batters long *and* with no variance at all,
in a league where 30.7% of starts fail to reach the fifth inning and only 37.6%
reach the sixth.

**Bullpen roster.** Who is actually available, inferred from recent
appearances rather than a season-long pool. The stale pooled-prior-season
feature this replaces had its regression coefficient collapse to zero whenever
Elo was present, which is direct evidence it carried no current information.

**Usage weighting.** Relievers do not split innings evenly. Team bullpen
quality is the usage-weighted projection of the arms likely to throw, not a
flat average over everyone who has appeared.

Bullpens cover 41.5% of all outs, so none of this is a refinement.
"""

import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

OUTCOMES = config.OUTCOMES

# Relief appearances within this many days define the active bullpen. Long
# enough to catch a low-leverage arm who has not pitched in a while, short
# enough to drop players who were optioned out.
ROSTER_WINDOW_DAYS = 21

# Below this many surviving arms, an active-roster filter is treated as bad data
# rather than obeyed. A real bullpen never has fewer; hitting this means the
# snapshot is stale or a team abbreviation failed to match, and simulating an
# almost-empty pen would be a worse error than simulating a stale one.
MIN_ACTIVE_ARMS = 5

# A reliever who pitched on each of the previous two days is treated as
# unavailable. Three straight days happens but is rare enough to ignore.
BACK_TO_BACK_LIMIT = 2

# Two entirely different things end a start early, and they need separate
# representation:
#
#   OPENERS are a property of the PITCHER, known before the first pitch. They
#   are relievers who happen to throw the first inning -- 3.07% of starts, with
#   a prior relief share of 0.93 against 0.14 for regular starters, and outings
#   (4.4 outs, 0.75 runs) statistically indistinguishable from ordinary relief
#   appearances (3.4 outs, 0.63 runs).
#
#   DISASTERS are a property of the GAME. Among starts under 12 outs recorded,
#   78.4% are genuine blowups by pitchers who normally work deep: 3.90 runs in
#   8.4 outs, a rate of 0.464 runs per out against 0.115 for a normal start.
#
# Measuring workload in BATTERS FACED hides this, which is what an earlier
# version of this model did. A shelled starter faces plenty of batters while
# recording few outs -- his batters-per-out runs 1.88 against 1.35 for a normal
# start -- so a batters-faced cutoff selects openers and buries disasters.
# Outs recorded separates them cleanly.
HOOK_FEATURES = [
    "outs_recorded", "pitches", "tto_next", "runs_allowed", "score_diff",
    "leading", "inning", "outs_after", "inning_ended", "runners_on",
    "bases_empty", "days_rest",
    # Planned workload, from the pitcher's own prior starts, in outs.
    "expected_outs", "is_opener",
    # Disaster signals: damage rate and traffic allowed, both per out, so they
    # fire early rather than waiting for a pitch count to accumulate.
    "runs_per_out", "traffic",
    # Piecewise terms on pitch count. Removal is not linear in pitches; it
    # accelerates sharply through the usual limits.
    "pitches_over_60", "pitches_over_80", "pitches_over_95",
    # Leash interactions. A workhorse having a good night is treated
    # differently from a short-outing pitcher getting hit, and additive terms
    # cannot express that. Measured over 2023+ starts reaching 80 pitches:
    # a workhorse who is cruising reaches 18 outs 78.5% of the time against
    # 31.9% for a short-leash starter being hit around, and roughly 3.6 points
    # of that gap is interaction rather than the two effects summed.
    "workhorse_x_pitches", "workhorse_x_damage", "pitches_x_damage",
    # Indicators by workload band. The hazard's shape in outs is not something
    # a linear term can follow: fitted without these it over-predicted removal
    # through the fifth inning (5.93% against an actual 4.93%) and
    # under-predicted it in the seventh, which pulled simulated starters about
    # half an inning short. These let each region find its own level.
    "outs_12_15", "outs_15_18", "outs_18_21", "outs_21_plus",
]


# --------------------------------------------------------------------------
# Hook model
# --------------------------------------------------------------------------

def starter_workload_history(pa, min_prior=2, league_default=15.5,
                             opener_outs=9.0, opener_relief_share=0.5):
    """Each start's expected OUTS, plus a pre-game opener flag.

    Both are strictly backward-looking (expanding means, shifted), so they are
    available before the first pitch.

    Outs rather than batters faced: a starter getting hit hard racks up batters
    without recording outs, so a batters-faced prior conflates "this man throws
    short outings by design" with "this man is getting shelled tonight". Only
    the first is knowable in advance, and only outs isolate it.

    The opener flag additionally requires that the pitcher works mostly in
    relief. Prior relief share separates the populations almost perfectly --
    0.93 for openers against 0.14 for regular starters -- so a workload
    threshold alone would sweep in bulk guys and rehabbing starters.
    """
    d = pa[pa["is_pa"]]
    g = (d.groupby(["game_pk", "game_date", "pitcher", "is_starter"])
         .agg(outs=("outs_pa", "sum")).reset_index()
         .sort_values(["pitcher", "game_date"]))

    # Prior share of appearances made in relief.
    g["relief"] = (~g["is_starter"]).astype(float)
    g["prior_relief_share"] = (g.groupby("pitcher")["relief"]
                               .transform(lambda s: s.shift(1).expanding().mean()))

    starts = g[g["is_starter"]].copy()
    starts["prior_outs"] = (starts.groupby("pitcher")["outs"]
                            .transform(lambda s: s.shift(1).expanding().mean()))
    starts["prior_n"] = starts.groupby("pitcher").cumcount()

    starts["expected_outs"] = np.where(starts["prior_n"] >= min_prior,
                                       starts["prior_outs"], league_default)
    starts["expected_outs"] = starts["expected_outs"].fillna(league_default)

    share = starts["prior_relief_share"].fillna(0.0)
    starts["is_opener"] = ((starts["expected_outs"] < opener_outs)
                           & (share > opener_relief_share)).astype(float)

    return starts[["game_pk", "pitcher", "expected_outs", "is_opener"]]


def hook_training_frame(pa, seasons=None):
    """One row per batter the starter faced, labelled with whether he was
    removed immediately afterward.

    The decision point is *between* plate appearances, so features describe the
    state the manager is looking at when he decides whether to send his starter
    back out.
    """
    d = pa[pa["is_pa"]].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]

    d = d.sort_values(["game_pk", "at_bat_number"])
    grp = d.groupby(["game_pk", "fielding_team"], sort=False)

    d["next_pitcher"] = grp["pitcher"].shift(-1)
    d["is_last_pa"] = d["next_pitcher"].isna()

    # Runs the starter has allowed so far, and his cumulative workload.
    d["fld_runs"] = d["runs_pa"].fillna(0)
    d["runs_allowed"] = grp["fld_runs"].cumsum()

    s = d[d["is_starter"]].copy()
    # Removed = the very next batter this team pitched to faced someone else.
    s["removed"] = ((s["next_pitcher"].notna())
                    & (s["next_pitcher"] != s["pitcher"])).astype(float)
    # Games that simply ended are not hook decisions and must not be labelled
    # as the starter surviving.
    s = s[~s["is_last_pa"]]

    s["bf"] = s["pitcher_bf"].astype(float)
    s["pitches"] = s["pitcher_pitches"].astype(float)
    # Outs the starter has actually recorded, the workload measure that
    # separates a short planned outing from a shelling.
    s["outs_recorded"] = (s.groupby(["game_pk", "fielding_team"])["outs_pa"]
                          .cumsum().fillna(0).astype(float))
    denom = s["outs_recorded"].clip(lower=3.0)
    s["runs_per_out"] = s["runs_allowed"] / denom
    s["traffic"] = (s["bf"] - s["outs_recorded"]).clip(lower=0) / denom
    # Times through the order the STARTER would be on for the next batter,
    # derived from his own workload. Reading it off the next plate appearance
    # instead leaks the answer: if he was pulled, that batter faces a new
    # pitcher whose counter has reset to 1, which made the feature a near
    # perfect predictor of removal (in-sample AUC 0.996).
    s["tto_next"] = np.floor(s["bf"] / 9.0).clip(0, 3) + 1.0
    s["outs_after"] = s["outs_after"].fillna(3).astype(float)
    s["inning_ended"] = s["ends_half"].astype(float)
    s["score_diff"] = (-s["score_diff_before"]).astype(float)  # from the pitcher's side
    s["leading"] = (s["score_diff"] > 0).astype(float)
    s["runners_on"] = (s[["r1_occ_after", "r2_occ_after", "r3_occ_after"]]
                       .fillna(False).sum(axis=1).astype(float))
    s["bases_empty"] = (s["runners_on"] == 0).astype(float)
    s["days_rest"] = s["pitcher_days_since_prev_game"].fillna(5).clip(0, 10).astype(float)
    s["inning"] = s["inning"].astype(float)

    for thresh in (60, 80, 95):
        s[f"pitches_over_{thresh}"] = (s["pitches"] - thresh).clip(lower=0)

    workload = starter_workload_history(pa)
    s = s.merge(workload, on=["game_pk", "pitcher"], how="left")
    s["expected_outs"] = s["expected_outs"].fillna(15.5)
    s["is_opener"] = s["is_opener"].fillna(0.0)

    # Centred so the interactions capture departures from a typical starter
    # rather than absorbing the main effects.
    wh = (s["expected_outs"] - 15.5) / 5.0
    pc = (s["pitches"] - 80.0) / 20.0
    dmg = s["runs_per_out"] - 0.115
    s["workhorse_x_pitches"] = wh * pc
    s["workhorse_x_damage"] = wh * dmg
    s["pitches_x_damage"] = pc * dmg

    o = s["outs_recorded"]
    s["outs_12_15"] = ((o >= 12) & (o < 15)).astype(float)
    s["outs_15_18"] = ((o >= 15) & (o < 18)).astype(float)
    s["outs_18_21"] = ((o >= 18) & (o < 21)).astype(float)
    s["outs_21_plus"] = (o >= 21).astype(float)

    return s[HOOK_FEATURES + ["removed", "season", "game_pk", "pitcher"]].dropna()


def fit_logistic(X, y, iters=1500, lr=0.5, l2=1e-4):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w, mu, sd


@dataclass
class HookModel:
    w: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    def hazard(self, X):
        """Probability the starter does not face the next batter."""
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-(Z @ self.w)))


def fit_hook_model(pa, seasons=None):
    fr = hook_training_frame(pa, seasons)
    X = fr[HOOK_FEATURES].to_numpy(dtype=float)
    y = fr["removed"].to_numpy(dtype=float)
    w, mu, sd = fit_logistic(X, y)
    return HookModel(w, mu, sd), fr


# --------------------------------------------------------------------------
# Bullpen rosters and quality
# --------------------------------------------------------------------------

def relief_appearances(pa):
    """One row per reliever per team-game: batters faced, date, entry inning.

    Entry inning is what orders the bullpen. Sorting by usage would send the
    closer out in the sixth, since the highest-usage arms are typically the
    late-inning ones -- the order a manager actually goes in is roughly by
    when each reliever normally enters.
    """
    d = pa[pa["is_pa"] & ~pa["is_starter"]]
    g = (d.groupby(["game_pk", "game_date", "season", "fielding_team", "pitcher"])
         .agg(bf=("is_pa", "size"), entry_inning=("inning", "min"))
         .reset_index())
    return g.sort_values("game_date")


class BullpenRosters:
    """Active bullpen and usage share for each team as of any date.

    Everything is derived from appearances strictly before the date in
    question, so it is available at the stated prediction point.
    """

    def __init__(self, appearances, window_days=ROSTER_WINDOW_DAYS,
                 rosters=None):
        self.app = appearances.copy()
        self.app["game_date"] = pd.to_datetime(self.app["game_date"])
        self.window = pd.Timedelta(days=window_days)
        self._by_team = {t: g.sort_values("game_date")
                         for t, g in self.app.groupby("fielding_team")}

        # Active-roster filter. The appearance window alone cannot tell an
        # injured pitcher from a rested one, so an arm placed on the IL kept
        # being simulated for up to `window_days`. Pass `rosters=False` to
        # disable; the default loads them if the file exists and otherwise
        # leaves behaviour exactly as it was.
        if rosters is False:
            self.rosters = None
        elif rosters is None:
            try:
                import active_rosters
                r = active_rosters.load()
                self.rosters = r if r.available() else None
            except Exception:
                self.rosters = None
        else:
            self.rosters = rosters

    _warned = set()

    def _warn_guard(self, team, date, n_active, n_total):
        key = (team, pd.Timestamp(date).normalize())
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"  NOTE: {team} on {key[1].date()} -- only {n_active} of "
              f"{n_total} recent relievers are on the active roster; keeping "
              f"the unfiltered pen (bullpen turned over, arms not yet modelled)")

    def as_of(self, team, date, with_entry=False):
        """Returns (pitcher_ids, usage_weights, available_mask), optionally
        with each reliever's mean entry inning."""
        empty = (np.array([]), np.array([]), np.array([], dtype=bool))
        g = self._by_team.get(team)
        if g is None:
            return empty + ((np.array([]),) if with_entry else ())

        date = pd.Timestamp(date)
        recent = g[(g["game_date"] < date) & (g["game_date"] >= date - self.window)]
        if len(recent) == 0:
            return empty + ((np.array([]),) if with_entry else ())

        agg = recent.groupby("pitcher").agg(
            bf=("bf", "sum"), entry=("entry_inning", "mean"))
        ids = agg.index.to_numpy()
        w = agg["bf"].to_numpy(dtype=float)
        entry = agg["entry"].to_numpy(dtype=float)

        # Drop anyone not on the active roster -- injured, optioned, or since
        # traded away. Their usage share is not discarded but redistributed,
        # because the innings still have to be pitched by somebody: `w` is
        # renormalised over the survivors below.
        #
        # A roster is only trusted when it leaves at least a workable pen. If
        # filtering would empty it, the snapshot is stale or the club's
        # abbreviation did not match, and an unfiltered bullpen beats none.
        if self.rosters is not None:
            act = self.rosters.as_of(team, date)
            if act:
                keep = np.array([int(p) in act for p in ids])
                if keep.sum() >= MIN_ACTIVE_ARMS:
                    ids, w, entry = ids[keep], w[keep], entry[keep]
                else:
                    # Say so rather than fall back quietly. This fires when a
                    # club has turned its bullpen over wholesale -- the 2026
                    # deadline Mets lost seven of eleven arms and replaced them
                    # with pitchers who had never relieved for them, so neither
                    # the stale list nor the four survivors describes that pen.
                    # The stale list at least has the right shape.
                    self._warn_guard(team, date, int(keep.sum()), len(ids))

        w = w / w.sum()

        # Availability: worked each of the last BACK_TO_BACK_LIMIT days.
        recent_days = {}
        for pid, grp in recent.groupby("pitcher"):
            recent_days[pid] = set(grp["game_date"].dt.normalize())
        prior = [(date - pd.Timedelta(days=k)).normalize()
                 for k in range(1, BACK_TO_BACK_LIMIT + 1)]
        avail = np.array([not all(p in recent_days.get(pid, set()) for p in prior)
                          for pid in ids])
        if with_entry:
            return ids, w, avail, entry
        return ids, w, avail


def bullpen_talent(model, roster_ids, weights, lw, available=None):
    """Usage-weighted expected run value allowed per plate appearance.

    Weighting by usage matters: a team whose best arm throws a fifth of the
    relief innings is meaningfully better than one whose innings are spread
    evenly, and a flat average erases that entirely.
    """
    if len(roster_ids) == 0:
        return None

    if available is not None and available.any():
        roster_ids, weights = roster_ids[available], weights[available]
        weights = weights / weights.sum()

    vals = []
    for pid in roster_ids:
        v = np.mean([np.dot(model.pitcher_vector(int(pid), hand, side), lw)
                     for hand in ["L", "R"] for side in ["L", "R"]])
        vals.append(v)
    return float(np.dot(weights, vals))


def bullpen_sequence(roster_ids, weights, entry_innings,
                     available=None, max_arms=8, min_usage=0.02):
    """The relief arms in the order a manager would use them.

    Ordered by mean entry inning, not by usage: the highest-usage relievers are
    typically the late-inning ones, so a usage ordering would send the closer
    out in the sixth. Low-usage arms are dropped -- a mop-up man who threw once
    in three weeks should not be third out of the pen.
    """
    if len(roster_ids) == 0:
        return np.array([])

    ids, w, entry = roster_ids, weights, entry_innings
    if available is not None and available.any():
        ids, w, entry = ids[available], w[available], entry[available]
        w = w / w.sum()

    keep = w >= min_usage
    if keep.sum() >= 3:
        ids, w, entry = ids[keep], w[keep], entry[keep]

    order = np.argsort(entry)[:max_arms]
    return ids[order].astype(int)


def main():
    import outcome_model as om

    pa = om.load_pa()
    print("fitting hook model on 2023-2025...")
    model, fr = fit_hook_model(pa, seasons=[2023, 2024, 2025])
    print(f"  {len(fr):,} decision points, "
          f"removal rate {fr['removed'].mean() * 100:.2f}%")

    print("\n  standardized coefficients:")
    for name, coef in zip(["intercept"] + HOOK_FEATURES, model.w):
        print(f"    {name:<14} {coef:+.4f}")

    p = model.hazard(fr[HOOK_FEATURES].to_numpy(dtype=float))
    y = fr["removed"].to_numpy()
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"\n  in-sample AUC {auc:.4f}")

    print("\n  hazard by outs recorded (average over other conditions):")
    fr = fr.assign(p=p)
    for lo, hi in [(0, 6), (6, 12), (12, 15), (15, 18), (18, 21), (21, 30)]:
        m = (fr["outs_recorded"] >= lo) & (fr["outs_recorded"] < hi)
        if m.sum():
            print(f"    outs {lo:>2}-{hi:<2}: predicted {fr.loc[m, 'p'].mean()*100:5.2f}%  "
                  f"actual {fr.loc[m, 'removed'].mean()*100:5.2f}%  (n={m.sum():,})")

    print("\n  hazard by damage rate (runs allowed per out, before 12 outs):")
    early = fr[fr["outs_recorded"] < 12]
    for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 5.0)]:
        m = (early["runs_per_out"] >= lo) & (early["runs_per_out"] < hi)
        if m.sum():
            print(f"    {lo:.2f}-{hi:<4.2f}: predicted {early.loc[m, 'p'].mean()*100:5.2f}%  "
                  f"actual {early.loc[m, 'removed'].mean()*100:5.2f}%  (n={m.sum():,})")

    print("\n  openers vs regular starters:")
    for lbl, m in [("opener", fr["is_opener"] > 0.5),
                   ("regular", fr["is_opener"] <= 0.5)]:
        if m.sum():
            print(f"    {lbl:<8} predicted {fr.loc[m, 'p'].mean()*100:5.2f}%  "
                  f"actual {fr.loc[m, 'removed'].mean()*100:5.2f}%  (n={m.sum():,})")

    print("\n  hazard by times through the order:")
    for t in [1, 2, 3, 4]:
        m = fr["tto_next"] == t
        if m.sum():
            print(f"    TTO {t}: predicted {fr.loc[m, 'p'].mean()*100:5.2f}%  "
                  f"actual {fr.loc[m, 'removed'].mean()*100:5.2f}%  (n={m.sum():,})")

    print("\nbullpen rosters:")
    rosters = BullpenRosters(relief_appearances(pa))
    ids, w, avail = rosters.as_of("LAD", "2025-07-01")
    print(f"  LAD as of 2025-07-01: {len(ids)} arms, {avail.sum()} available")
    print(f"  usage share of top 3: {np.sort(w)[::-1][:3].round(3).tolist()}")


if __name__ == "__main__":
    main()
