# Findings

What was tried, what the numbers were, and what shipped. The negative results
are the more useful half — several cost real time to establish and would
otherwise be re-attempted.

All figures are holdout 2025, 2,430 games, unless stated.

---

## The simulator's progression

| stage | AUC | log loss | note |
|---|---|---|---|
| fixed usage, no defence/speed | — | 0.68337 | starter pulled at 24 batters, flat |
| + defence and baserunning | 0.55990 | 0.68436 | **worse** |
| + bullpen and hook models | 0.57190 | 0.68040 | largest single gain |
| + dispersion (σ 0.19) | 0.57149 | 0.68097 | trades sharpness for a calibrated distribution |
| + opener/disaster split | 0.57149 | 0.68097 | fixed short starts |
| **+ team expansion ×1.2** | **0.57582** | **0.67960** | shipped |
| blended with Elo, w(sim) 0.4 | **0.58141** | **0.67700** | shipped |

---

## What shipped

**Bullpen and hook models — the largest single gain.** Bullpens cover **41.5% of
all outs**, not the third assumed. Median start is 5.33 innings; only 37.6%
reach six. The fixed rule pulled every starter at 24 batters faced against a
real mean of 21.87 with a standard deviation of 4.65 — late, and with no
variance at all. A logistic removal hazard (pitch count dominant, then inning
boundary) plus roster-aware bullpens **ordered by mean entry inning rather than
usage** — the highest-usage arms are the late-inning ones, so a usage ordering
sends the closer out in the sixth.

**Home field advantage had to be added explicitly.** Without it the simulator
produced a 50.13% home win rate against an actual 54.03% and was *less* accurate
than always picking the home team. It is a real per-plate-appearance effect —
home batters strike out 3.6% less, walk 6.8% more, homer 4.6% more — and it is
easy to miss because runs *per game* look flat, an artifact of the home team
skipping the bottom of the ninth when ahead.

**Dispersion, drawn per team rather than shared.** Home/away run covariance is
−0.011 (correlation −0.0011) — essentially zero. A shared shock (weather,
umpire) would create positive covariance and there is none to create. Must be
mean-centred, or the 4% run inflation from `exp(σ²/2)` gives the right spread
and the wrong mean. Team-runs std ratio 0.9077 → 0.9941.

**Openers and disaster starts, separated.** Measuring workload in *batters
faced* conflates them: a shelled starter piles up batters while recording few
outs (1.88 per out against 1.35 normal), so a batters-faced cutoff selects
openers and buries disasters. Cutting on **outs recorded**, 78.4% of starts
under 12 outs are genuine blowups (3.90 runs in 8.4 outs), not openers. Openers
are a *pitcher* property known pre-game (prior relief share 0.93 against 0.14);
disasters are a *game* property needing in-game damage-rate features. Simulated
short-start share 1.86% → 5.03% against 4.45% actual.

**Team-strength expansion ×1.2.** Regressing actual team run differential on
predicted gives a slope of 1.46 — the simulator understates team differences by
about 45%, exactly what summing individually-shrunk players produces.

---

## What did not ship, and why

### Defence and baserunning helped the logistic, not the simulator

Per-player OAA and sprint-speed-conditioned transitions. The measured effects
are large — scoring from first on a double runs 21% for the slow tercile against
46% for the fast; double-play rate 27.4% against 19.1% by batter speed.

Result: the talent logistic improved (0.68165 → 0.68065) and the **simulator got
worse** (0.68337 → 0.68436), with the dispersion ratio completely unmoved.

A logistic fits coefficients to outcomes; the simulator applies them structurally
at magnitudes derived from first principles and cannot self-correct. The
components were kept because they are needed for the decision layer.

### Platt calibration recovered 7% of the gap

Fitted on 2024, applied to 2025: log loss 0.68436 → 0.68401, **AUC bit-identical
at 0.55990** — the expected confirmation that a monotone transform cannot
reorder anything.

Of the 0.00500 gap to Elo, calibration explained 0.00035. **93% is
discrimination.** Elo's Spearman correlation with actual team run differential
is 0.8855 against the simulator's 0.8005, and Spearman is scale-immune, so this
is accuracy rather than compression.

### Elo talent-layer injection — the most instructive failure

Blending Elo-derived team strength into the talent layer. AUC 0.57582 →
**0.56753**; the blend degraded from 0.58141 → 0.57550.

The mechanism fired exactly as designed: predictions moved (mean absolute change
0.029), correlation with Elo's log-odds rose 0.665 → **0.799**, spread widened,
compression improved 1.458 → 1.359.

**And that correlation is why it failed. The simulator's value in the blend was
its independence from Elo.** Injection made it a noisier copy of Elo — worse
alone and far less complementary combined.

The same trade explains why expansion ×1.45 loses to ×1.20 despite ×1.45 nearly
eliminating the compression it was derived from (slope 1.0426 vs 1.2443).
Correlation with Elo runs 0.6650 / 0.7019 / 0.7394 across ×1.00 / ×1.20 / ×1.45.
**The measured 1.46 slope is a diagnostic, not an objective.**

### Constant tuning won at plate-appearance level and lost at game level

Leave-one-season-out over 2023–2025, staged so a shared multiplier is fitted
before any individual constant. Plate-appearance log loss 1.47459 → 1.47338
(+0.082%), winning all three folds.

The staging earned its keep: the shared multiplier explained the whole gain and
all sixteen individual constants added +0.00011 combined.

In the simulator: AUC 0.57582 → **0.57011**, compression slope 1.2443 →
**1.4468** — almost exactly the 1.4579 the expansion was built to fix. The
tuner wants 3–4× more player shrinkage; more shrinkage narrows projected player
spread; narrower spread widens team compression. **Heavier player shrinkage
silently undid the team expansion.**

The structural tension worth keeping: **at the player level more compression is
correct** (the right response to sampling noise in one record); **at the team
level, aggregating compressed players under-disperses team strength.** Under
*random* player assignment shrink-then-sum would be optimal — it is the positive
correlation of talent within rosters that makes the aggregate deserve less
shrinkage than its parts.

### Substitution modelling, rejected before building

Only **3.74%** of plate appearances go to substitutes (1.41 per team-game,
concentrated late — 14.1% in the ninth). Across 5,116 swaps the incoming batter
is worth **+0.00116 runs per plate appearance**, with 46% being downgrades.

Implied effect: **+0.0016 runs per team-game** — one thirty-fifth of the team
defence effect, which itself moved nothing. Managers' substitutions are
individually valuable (+1 to +3 points of win probability) and collectively
worth almost nothing, because gains and losses cancel.

---

## Bugs found, all silent

Every one produced plausible output while being wrong.

1. **Shared shrinkage denominator** — per-outcome regression divided by
   `n + sum(k)` instead of `n + k_o`, producing a model *worse* than a league
   constant while predicting 13% doubles against an actual 4.1%.
2. **Double-counted platoon** — the odds ratio divided by the hand-agnostic
   league vector while both sides were already platoon-adjusted.
3. **Lost runs, two causes** — state-moving rows with no event were filtered
   out, and Statcast's `post_bat_score` sometimes omits a run scored on that
   same play. Fixed by deriving runs from consecutive *entry* scores, which
   telescopes exactly.
4. **`GroupBy.first()`** returns the first non-null value of each column
   independently, not the first row. A runner stealing third mid-at-bat
   populated `on_3b` for the plate appearance's *entry* state, corrupting every
   base-out state. Bases-loaded home runs averaged 3.82 runs instead of 4.00.
5. **`is_starter` only matched home starters** — it compared each pitcher's
   first at-bat to the *game's* first at-bat, but the away starter debuts in the
   bottom of the first. Every away starter was classified as a reliever.
6. **Hook model leakage** — `n_thruorder_pitcher` read from the *next* plate
   appearance, but a pulled starter's replacement has a counter reset to 1, so
   the feature encoded the label. In-sample AUC 0.9962, with removals occurring
   at "times through order 1" and never at 2, 3 or 4.
7. **Team ordering in the Elo injection** — returned `(home, away)` where the
   simulator indexes `(away, home)` throughout, applying each team's correction
   to its opponent.
8. **Conditional random draws** — two draws in `simulate()` fired only when a
   condition held, desynchronising common-random-number branches.

Bugs 4, 6 and 8 were caught by results being implausible rather than by any
test. Bug 6 in particular looked like a triumph.

---

## Process notes

**Measure before building.** The substitution rejection cost twenty minutes; the
Elo injection cost an afternoon to reach the same kind of answer. Cheap
measurement first would have caught more of them.

**Do not read a subsample.** The Elo strength sweep was called a "clear negative
result" from three cells of a 600-game subsample where every difference was
under 0.002 nats against an accuracy standard error near 2 points. The unseen
fourth cell was the best accuracy the simulator had produced. The full-season
run agreed with the conclusion but not for the stated reasons.

**Component gains do not transfer.** Roughly half the changes that improved
their own metric made the simulator worse. Verify end to end, always.

## Does the simulator know anything the closing line does not? (2026-08-03)

Blending against the market tests *residual* information: conditional on
everything the line prices, does our number move it the right way? A model can
be worse than the market on every metric and still improve it, if its errors are
uncorrelated with the market's. That is the only sense in which this project
could have edge.

Sample: 52 games (2026-07-29 to 08-01) with a recorded closing line, a
simulator prediction and a result. Captured a median 114 min before first pitch.

**Simulator: clean null.** Every blend weight makes the market worse,
monotonically (w=0.05 costs +0.00061 nats, w=0.50 costs +0.00801). Nothing
residual detected. P(blend beats market) = 0.26.

**Elo: an artifact, not a result.** The blend appears to improve the market,
but improvement continues monotonically to w=1.00 -- the "optimal blend"
discards the closing line entirely. That is the reductio: Elo merely posted a
lower log loss than the market on these 52 games by chance. Across all 1,667
2026 games Elo's log loss is 0.68959 and the simulator's 0.69247, against the
market's 0.66954. Neither is close.

Three diagnostics were wrong on the first pass and are fixed in `market_blend.py`:

- `P(blend beats market)` was computed with the inequality reversed.
- Correlating raw residuals `y - p` is meaningless when every prediction sits
  near 0.5, because `y - p` is dominated by the shared `y`; it read +0.99 for
  both models regardless of independence. Log-odds prediction correlation is
  the real diagnostic: simulator +0.719, Elo +0.750.
- A shrinkage control was missing. It matters: shrinking the market toward 0.5
  *hurts* on this sample, which rules out the obvious explanation that an
  under-confident model was helping by adding timidity.

**Where this leaves the project.** The closing line contains what both models
know about who wins. That is the expected state of a liquid market, not a
failure. Remaining plausible value is in totals and props, where the market is
thinner and the run-level model has an actual mechanism -- not moneylines.

**Caveat.** n=52 resolves roughly 0.14 SE against effects of order 0.005 nats.
The simulator null is clean in *direction* (wrong sign at every weight), which
is more informative than its significance. Re-run when the recorder has a few
hundred games.

## Pitch level, phase one: arsenal fit vs strikeout calibration (2026-08-03)

Refetched Statcast with location, movement, arm slot and spin (99.6% coverage)
plus bat tracking (46.2%, 2025+). Built `pitch_*` tables: arsenal mix, and
whiff-per-swing by pitch group for every batter and pitcher.

**The arsenal hypothesis fails.** Tested whether a pitcher's specific mix
attacking a specific hitter's specific hole predicts strikeouts beyond both
players' overall rates. The feature is one number -- a log ratio of the
mix-weighted per-group whiff expectation against the same expectation built
from overall rates, so both main effects cancel and only the interaction
survives.

Held out on 2025 it improves PA-level strikeout log loss by 0.000126 nats
(5.6 sigma on 183k PAs). But a permutation control -- reassign batters at
random, which destroys real matchup fit while leaving arsenal type intact --
recovers 57% of that gain. The genuine interaction remnant is 0.000054 nats per
PA, far too small to move a prop. Batter pitch-type vulnerabilities do not
stabilize enough to be worth modelling.

An earlier version of the feature was worse than useless: it compared a
mix-weighted expectation against an unweighted overall one, which are not on the
same scale, so it tracked arsenal *type* and came out anti-correlated with
strikeouts. Fixed by expressing each player's per-group rate relative to his own
overall rate.

**The calibration finding is the valuable one.** Strikeouts are over-predicted
in every season -- 2022 +2.0%, 2023 +0.8%, 2024 +1.8%, 2025 +4.0%, 2026 +2.6%
-- and consistently worse for starters (+2.5% to +5.2%). The error grows with
the prediction: on 2025 the lowest bucket was under-predicted 2.8% while the
highest was over by 10.1%. That is a slope bias, not a level bias, so
`sim_calibration.json` (one multiplier per outcome) cannot touch it.

Fitted `logit(p') = a + b*logit(p)` on 2024: a = -0.178, b = 0.868, i.e. the
strikeout spread is 13.2% too wide. Held out:

- 2025: top bucket +10.1% -> +3.6%; starter 24-BF bias +0.24 -> +0.15 K
- 2026: top bucket  +5.0% -> -1.1%; starter 24-BF bias +0.13 -> +0.05 K
- log loss improves on both

Cost: the lowest bucket gets worse (2025 +1.4% -> +8.4%). This trades accuracy
on low-strikeout matchups for accuracy in the tail. Correct for props, wrong if
K probabilities are used for anything else.

**Not yet wired into props, deliberately.** The constant was fitted using models
built from strictly prior seasons, but `PropSimulator` refreshes monthly with
season-to-date counts. The bias under that configuration will be smaller and
must be re-measured before the correction is applied, or it will over-correct.

### Correction: the "hook shortfall" was a stale artifact (2026-08-03)

While wiring in the strikeout calibration I read `prop_pitching_2025.csv` and
found starters projected 15.12 outs against 15.66 actual, and built a story on
it: that a hook pulling starters half an out early was cancelling the strikeout
rate bias, and that correcting the rate alone would deepen an unders lean.

That file was dated 29 July -- before the `sim_calibration` patch and before the
Savant park factors landed. A live sweep at `hazard_scale` 1.00 gives 15.560
projected against 15.438 actual on the same games: the hook runs slightly
*long*, not short. There is no shortfall, no cancellation, and no hook
correction is warranted. `hazard_scale` stays at its default of 1.0; the lever
is left in `HazardHook` because it is cheap and the sweep is now reproducible.

The strikeout rate bias itself is unaffected -- it was measured directly from PA
data without the simulator, so no stale artifact touches it.

Lesson: check artifact timestamps against the changes they were meant to reflect
before reasoning from them. Two of this project's larger changes landed the same
morning.

### K calibration wired into props: A/B verdict (2026-08-03)

Refitted under season-to-date to match `PropSimulator` (a = -0.109, b = 0.928 --
a 7.2% pull-in, against 13.2% from the prior-seasons fit, confirming the earlier
constant would have over-corrected nearly twofold). Wired into
`simulate.build_cum_probs` via `_recalibrate_k`, applied after normalisation:
the strikeout share is recalibrated in log-odds and the seven other outcomes are
rescaled by a common factor, preserving their proportions.

A/B on 640 starts, identical games and seed:

- mean bias +0.071 -> -0.028 K per start (systematic, not noise)
- MAE +0.00430, 95% CI [-0.00712, +0.01570], P(better) 0.77
- MSE +0.01953, 95% CI [-0.03929, +0.07777], P(better) 0.75
- over/under log loss: 3.5 -0.00285, 4.5 -0.00338, 5.5 +0.00047, 6.5 +0.00029

**Correction to the earlier framing in this file.** The claim that the bias is
concentrated where strikeout props live conflated two different slicings. It
holds for *P(K) buckets* at PA level. It does not hold for *projected K totals*,
which is what a line is posted on: the 6.5+ band was already unbiased (-0.03)
and the correction moves it to -0.31. With n=34 and RMSE 2.18 the SE there is
about 0.37, so that is noise -- but it is not an improvement on aces.

**Bound on the whole exercise.** RMSE is 2.18 against a mean of 4.64 strikeouts.
Start-level strikeout totals are dominated by irreducible variance, and no
recalibration of this shape moves that. Kept because it removes a demonstrated
systematic bias and improves held-out PA log loss on two seasons, not because it
buys measurable accuracy at the high lines.

Verify with: `python props.py --season 2025` and compare against
`prop_pitching_2025_nokcal.csv`.

### The unders lean, measured (2026-08-03)

Ran `check_prop_lean.py` on the 29 games that have both a recorded ESPN line and
a played result.

**First run was contaminated and has to be discarded.** The filter matched
prop_type on the substring "Strikeout", which catches two incompatible markets:
`Total Strikeouts` (the over/under) and `Strikeouts Thrown Milestones` (a yes/no
on reaching a threshold). Milestone numbers are structurally higher than the O/U
line for the same pitcher -- Skenes on 2026-07-31 was 8.5 O/U against a 9.0
milestone -- and `keep="last"` preferentially selected the milestone row because
those refreshed later. That single mismatch moved the mean line 4.85 -> 5.47,
the edge -0.19 -> -0.83, and the apparent win rate 62.7% -> 72.3%. It
manufactured an edge from a market-definition error.

Corrected, on 51 matched starts (96% of the lines whose games were simulated):

- mean line 4.85, model 4.67, mean edge -0.19
- model picks under 33 of 51 (65%) -- the observed lean is real but modest
- model's side won 32 of 51 (62.7%), zero pushes
- actual K 4.75, against a 4.85 line and a 4.67 model

**62.7% is not evidence.** SE at n=51 is 7.0 points, so this is 1.5 sigma over
the 52.4% break-even, on a sample examined after the fact. The lines were close
to fair (0.10 above what happened) and the model's 0.08 shortfall is well inside
noise.

Note against the calibration shipped the same day: it pulls projections down
about 0.10, so uncalibrated the model would have projected ~4.77 against 4.75
actual -- marginally closer than the calibrated 4.67. Meaningless at n=51, but
it is the direction to watch, and it is a reason not to assume the 2024 fit
transfers cleanly to 2026.

## ESPN collection outage, 2026-08-04

Symptom: no lines collected all day; projections fine. The scheduled task was
healthy throughout -- ran hourly, exit code 0, zero missed runs -- and the log
read `15 games, 0 matched to ESPN`, which looks like a quiet slate.

Cause: ESPN began returning **403** to the User-Agent this project sent,
`Mozilla/5.0 (research; personal use)`. `get_json` swallowed the status,
returned None, and the caller reported zero matches. Both moneylines and props
died at the same instant (2026-08-03 23:14 UTC), which is what ruled out a
props-specific or name-matching cause.

Tested systematically: ESPN accepts requests' own default (`python-requests/2.x`)
and 403s *every* custom User-Agent, including honest descriptive ones. The old
value was a malformed browser string and tripped a spoof heuristic. Fix is to
send no custom UA at all -- the library's own identifier is both the truthful
option and the one that works. Baseball Savant accepts all variants, so
`fetch_park_factors.py` was unaffected (it still carries the same string, which
is a latent risk if Savant ever adds the same rule).

Two hardening changes so this cannot fail silently again:

- `get_json` prints `FETCH FAILED (HTTP 403): <url>` when every attempt fails.
- Scheduled games with zero ESPN matches now emit an explicit WARNING naming it
  as a feed or identifier change rather than an empty slate.

Verified: 15 games, 15 matched; props log 71,740 -> 81,017 rows.

Lesson: an exit code of 0 from a scheduled collector proves the process ran, not
that it collected anything. Failures that degrade to "found nothing" are
invisible to the scheduler and need to be loud in the log.

### Park factor fetcher hardened (2026-08-04)

`fetch_park_factors.py` carried the same `Mozilla/5.0 (research; personal use)`
string that ESPN began 403ing. Savant accepts every UA variant, so this was
hardening rather than a fix, but it removes a known-flagged string from the last
place it appeared. `UA = None` there too.

Re-fetched and diffed against the pre-change file: seasons 2021-2025 are
identical to the digit, and only the 2026 row moved (20 index points total
across 15 parks, max 3). That is Savant recomputing its live 2024-2026 rolling
window after a day of games, not an artifact of the header -- a fetch difference
would have moved closed seasons too, and index_so/index_bb match exactly.

No effect on the simulator: `load(season)` reads the `season - 1` row, so 2026
projections use the unchanged 2025 window.

## Active-roster filter for bullpens (2026-08-04)

The simulator inferred reliever availability purely from appearances in the last
21 days, which cannot distinguish an injured pitcher from a rested one. An arm
placed on the IL kept being simulated for up to three weeks.

`active_rosters.py` fetches StatsAPI `rosterType=active` (status "A" only -- the
40-man view of the same club shows the D10/D15/D60 and minor-league
reassignments being excluded). The endpoint honours a `date` parameter, so
historical rosters are available too: a club's June 1 and August 4 active
rosters shared only 15 of 26 players. Backtests get the same correction once
backfilled, rather than being quietly more optimistic than live.

Positions are deliberately **not** used for classification. Every pitcher comes
back labelled "P"; the feed cannot separate starters from relievers and is never
asked to. Roles stay behavioural via `relief_appearances`. The roster answers
only "was he on the roster".

Measured on 2026-08-04: league bullpens go from 357 simulated arms to 245. All
eight spot-checks of removals were genuine ("Reassigned to Minors" or off the
40-man entirely); zero false positives. 23.8% of bullpen usage weight was being
assigned to pitchers who were not on the roster, and it is now redistributed
across arms that can actually take the ball.

Two guards, because a silent failure here would be worse than the staleness:

- `MIN_ACTIVE_ARMS = 5`: if filtering would leave fewer, the unfiltered pen is
  kept and a NOTE is printed naming the team. Fires for exactly one club --
  the deadline Mets, who lost seven of eleven arms and replaced them with
  pitchers who had never relieved for them.
- A missing or pre-snapshot roster file returns None and leaves behaviour
  exactly as it was. Verified: a 2025 date returns an identical bullpen with
  the filter on or off.

**Known gap.** The filter only prunes. It cannot add a newly acquired or
called-up arm, because he has no appearance history with the new club -- seven
active Mets pitchers have never relieved for them. Padding the pen with such
arms would work (the outcome model regresses an unseen pitcher to league average
on its own), but it changes bullpen composition materially and is unverified.

## Pinch hitting is absent from the bulk simulator

`build_cum_probs` assembles a fixed nine-slot lineup; batters never change over
a simulated game, only pitchers turn over. `decision_engine.py` does model
pinch hitters, but it is a separate tactical layer wired only into
`demo_engine.py`. The two halves of the project disagree about whether
substitutions exist.

Size of the gap, 2025:

- 3.74% of all plate appearances are taken by a substitute
- concentrated late: 2.7% in the 6th, 9.9% in the 8th, 14.1% in the 9th,
  18.6% in extras
- 62.8% of team-games feature at least one substitute PA, mean 1.41 per game
- **11.0% of lineup slots finish the game with a substitute in them**

For win probability this is second-order and the omission is defensible. For
hitter props it is not: a starting hitter loses his final plate appearance about
11% of the time, so the simulator overstates his PA count by roughly 2.6% and
inflates every counting prop (hits, total bases, home runs) by about that much.
That is the same order as the strikeout slope bias corrected earlier today.

This is unfinished work rather than a design decision. The blocker was bench
availability data, which `active_rosters.py` now provides (it carries position
players with an `is_pitcher` flag).

## Pinch hitting in the bulk simulator (2026-08-04)

`cum_probs` gained a trailing substitution axis (0 = the slot's starter, 1 =
after a pinch hitter enters), and the loop tracks per-slot substitution state.
A substitution is permanent for the slot. `pinch=None` reproduces the previous
fixed-lineup behaviour exactly, so nothing changes unless it is switched on.

**When** -- a discrete-time hazard evaluated every plate appearance, fitted on
2021-2025 (`pinch_hitting.py`), drawn unconditionally so common random numbers
stay aligned:

    intercept -6.16   late +0.95   very_late +0.33   log_li -0.0001
    platoon_disadv +0.84   trailing_close +0.42   pitcher_slot_gt0 +1.88

Calibration by inning is good: 0.35% predicted vs 0.00% actual in the first,
5.88 vs 5.28 in the seventh, 8.92 vs 8.85 in the ninth. Deciles are monotone.

**Leverage turned out not to matter.** Its coefficient is -0.0001: once the
model knows the inning and whether a reliever is already in, leverage adds
nothing to pinch-hitting decisions. That is the opposite of the pitching hook,
where it carried real signal. The term is retained (the table is built and it
costs nothing) but it is doing no work, and the honest description of this
model is "late innings and reliever presence", not "leverage-driven".

**Who** -- the league-average pinch hitter by platoon state, measured from real
substitute plate appearances rather than assumed. Bench bats are genuinely
worse: 27.3% strikeouts against a 22.6% league rate, and 2.4% home runs against
3.1%. The platoon advantage they enter with does not compensate.

Two fitting errors found and fixed along the way, both of which produced
confident wrong answers:

- The first target was "the *next* PA in this slot is a substitute", which
  attributes the decision to the starter's previous time up -- often innings
  earlier -- and censors substitutions where the slot never bats again. It
  fitted -2.22 on `very_late` and put ninth-inning substitutions at 0.35% when
  the truth is 14.1%. Fixed by attaching the event to the PA where the
  substitute actually bats, on the correct risk set.
- `platoon_disadv` was computed from the batter who appeared. On event rows
  that is the *substitute*, who was chosen for the platoon advantage, so the
  model learned that substitutions happen when the matchup is already good --
  a coefficient of -1.04 on the term that should be positive. Fixed by using
  the incumbent's handedness.

Verified end to end: 4.10% of simulated plate appearances are taken by
substitutes against a real 3.74%, and 0.00% with the feature off. The ~10%
relative overshoot is not yet corrected; it is worth about 0.36% of plate
appearances and should be trimmed with an intercept shift once there is a run
budget to re-verify it.

### Hitter prop baseline before pinch hitting (2026-08-04)

Preserved as `prop_hitting_2025_nopinch.csv` (K calibration on, substitutions
off) so the effect of pinch hitting is measured against a control that differs
in exactly one thing. On 5,760 batter-games:

    total bases    predicted 1.4936  actual 1.4443  (+0.0493, +3.4%)
    hits+runs+RBI  predicted 1.8797  actual 1.7988  (+0.0809, +4.5%)
    fantasy        predicted 6.6542  actual 6.4026  (+0.2516, +3.9%)

Every counting line is over-projected, in the direction and roughly the
magnitude predicted from the plate-appearance argument: a starter who is lifted
for a pinch hitter keeps batting in a simulator that has no bench.

### Pinch-hit intercept calibrated (2026-08-04)

The fitted hazard is well calibrated per plate appearance, but the quantity that
matters is the share of PAs *taken* by a substitute, and the two differ: once a
slot turns over it stays turned over, so the observed share depends on when
substitutions fire, not only how often. `calibrate_pinch.py` measures and
corrects rather than solving it analytically, holding games and seed fixed
across iterations so only the intercept varies.

    iter 1: intercept -6.1585 -> 4.096%  (target 3.742%)
    iter 2: intercept -6.2488 -> 3.768%  (+0.026 pp, stopped)

Every other coefficient is untouched; `intercept_fitted` is retained in the JSON
so the original per-PA fit can be recovered. Backup at `pinch_hazard.json.bak`.

### Hitter props with pinch hitting: partial fix (2026-08-04)

640 paired batter-games, identical games and seed, K calibration on in both.

    total bases    actual 1.4443   off +0.0493 (+3.4%)   on +0.0361 (+2.5%)
    hits+runs+RBI  actual 1.7988   off +0.0809 (+4.5%)   on +0.0649 (+3.6%)
    fantasy        actual 6.4026   off +0.2516 (+3.9%)   on +0.1983 (+3.1%)

MAE improves on all three (total bases 1.3487 -> 1.3439), so this is a real
improvement, but it recovers only about a quarter of the bias -- not the bulk,
as predicted beforehand.

**Why, and it is a known limitation rather than a mystery.** The simulator's
per-batter counters are indexed by *lineup slot*, not player id, and `props.py`
maps each slot back to the starting batter. So a pinch hitter's production is
still credited to the starter he replaced. Only half the mechanism is therefore
captured: the slot correctly switches to the (worse) pinch-hitter outcome rates,
which is the bias that did close, but the starter still receives the plate
appearance, so the PA-count effect -- the larger half, and the basis of the
2.6% estimate -- is untouched.

**Remaining fix:** stop accumulating into a slot's batter counters once
`sub_ax == 1`, so a substituted starter's line ends where it really ended. A
contained change to the tracking block in `simulate()`. Until it lands, hitter
counting props remain over-projected by roughly 2.5-3.6%, and that residual
should be treated as a known bias, not as noise.

### Slot attribution fixed -- hitter props now unbiased (2026-08-04)

The per-batter counters are indexed by lineup slot, not player id, so a pinch
hitter's production was landing on the starter's line and the starter kept
accruing plate appearances he never took. Two changes:

- The batter's own counters (hits, walks, RBI, self-scored runs) accumulate only
  while `sub_ax == 0`, so a substituted starter's line ends where it ended.
- A `base_sub` array carries whether each baserunner reached as a substitute,
  through the same canonical mapping as slot and speed. `base_slot` records only
  which *slot* is on the bag, so without it a pinch hitter's run was credited to
  the starter. Bases clear each half-inning, so it only has to be right within
  an inning.

Three-way, 5,760 paired batter-games, identical games and seed:

    total bases    actual 1.4443   no pinch +3.4%   bad attr +2.5%   fixed -0.6%
    hits+runs+RBI  actual 1.7988   no pinch +4.5%   bad attr +3.6%   fixed +0.2%
    fantasy        actual 6.4026   no pinch +3.9%   bad attr +3.1%   fixed -0.2%

MAE improves at every step (total bases 1.3487 -> 1.3439 -> 1.3314; fantasy
5.1842 -> 5.1641 -> 5.1072), so this is not a level shift that happens to zero
the mean -- the per-game predictions are genuinely better.

Independent confirmation of the target from `pa_events`, without the simulator:
league total bases are 13.60 per team-game, 1.511 per lineup slot, 1.464 per
starter-game. The gap between the per-slot and per-starter figures is exactly
the substitute production the fix removes.

**Correction to an earlier comment in this file.** The `base_sub` array was
justified with an example about a runner who singled in the seventh still being
aboard in the ninth. That is incoherent -- bases clear every half-inning. The
real reason is the within-inning case: a pinch hitter who reaches and scores.

## Stuff+ and archetype tests (2026-08-04)

2.1M pitches with full physics, 2023-2025. All three tested against a control.

**Stuff+ proxy: adds nothing.** Whiff-per-swing fitted from physics alone
(velocity, both movement planes, spin, arm angle, plate location), no pitcher
identity or outcome history, aggregated to pitcher-season. Against 676
pitcher-season pairs, last season's *rate* correlates +0.5906 with this season
while last season's *stuff* correlates +0.1795, and adding stuff to past rate
gains R2 +0.0010 with a *negative* weight (-0.0365).

**Caveat that matters: this is a weak Stuff+ proxy, not a refutation of the
idea.** Public Stuff+ models are gradient-boosted, fitted per pitch type, and
use features relative to the pitcher's own fastball. A ten-feature logistic
correlating 0.18 with future performance is well below the ~0.5 those report.
The honest claim is "our crude version adds nothing", not "stuff does not work".

**Velocity archetype: fails.** Real interaction -0.012409 nats, but shuffling
the batters recovers -0.010562 -- 85% survives. That is a velocity *main*
effect (league whiff/swing 29.7% in the top velocity bucket against 18.3% in the
bottom), which the model already carries through pitcher strikeout rates. No
batter-specific hole.

**Arm slot: passes the control.** League whiff/swing is flat across arm-slot
buckets (23.2 / 23.4 / 23.3%), so there is no main effect to leak. Real
interaction -0.001442, shuffled -0.000421: 71% of the gain dies when batters are
reassigned. This is the first feature tested in this project that survives its
own control.

But the magnitude is small -- a tenth of the velocity effect -- and it is
measured per swing on whiff, not per plate appearance on strikeouts, so it has
two translations to survive before it could move a prop. Three permutation reps
is also thin. Worth a proper test, not worth building on yet.

### Arm slot, tested properly (2026-08-04)

PA-level target, batter rates fitted on 2023-2024 and tested on 2025 which they
never saw, ten permutation reps. The earlier per-swing version fitted and
evaluated on the same rows, which inflates precisely the batter-specific effect
being claimed.

    182,840 held-out PAs
    baseline log loss  0.521367
    + arm-slot tilt    0.521353   (-0.000014)   beta +0.1694
    batters shuffled (10 reps): -0.000004, sd 0.000003
    28% of the gain survives the shuffle

**It passes the control again** -- 72% of the gain dies when batters are
reassigned, so the interaction is genuine. But out of sample at plate-appearance
level the effect is **-0.000014 nats**, roughly a hundredth of what the
in-sample per-swing test suggested (-0.001442) and about four times *smaller*
than the arsenal remnant already rejected as too small to matter (-0.000054).

Practical size: beta 0.1694 on a tilt with sd 0.0795 moves P(K) by about 0.23
percentage points per standard deviation, or 0.055 strikeouts across a
24-batter start. A two-sigma matchup moves a projection by a tenth of a
strikeout.

So: real, and the only feature in this project to survive its own control twice
-- but far too small to move a prop, and not worth the complexity. Not built.
The useful conclusion is methodological: an in-sample per-swing measurement
overstated a real effect by two orders of magnitude.

## Yahoo daily fantasy build (2026-08-22)

Yahoo's DFS API is open and unauthenticated. `contestsFilteredWeb` lists
contests with salary caps, `contestPlayers` returns the pool with salaries,
eligible positions, confirmed batting order and Yahoo's own `projectedPoints`,
and the series payload carries a **per-game weather forecast** -- wind,
temperature, precipitation -- which this project had no other source for.

**Scoring.** Hitters recovered from the feed: every game-score is an exact
combination of {2.6, 1.9, 4.2}, 400 of 400, pinning 2.6 per total base and per
walk, 1.9 per run and per RBI, 4.2 per steal. Pitchers could not be recovered
the same way -- `fantasyPointsHistory` holds the last ten *team* games, so a
starter's entries are mostly days he did not pitch -- and were supplied from
Yahoo's rules: win 4, K 2, out 1, earned run -2, hit/walk/HBP -0.9. Those
reproduce 60.9% of listed scores with a median difference of 0.00, against
5.6-9.6% for five guessed alternatives.

**Optimizer.** Exact MILP (`scipy.optimize.milp`, HiGHS) over the whole board,
with an assignment variable per (player, slot) so multi-eligible players are
considered at every position they qualify for. Overlap constraints diversify
multiple lineups; projection jitter explores near-optimal alternatives rather
than reshuffling fill order. Roster is 2 P, C, 1B, 2B, 3B, SS, 3 OF at a 200
cap.

**Eligibility.** Yahoo's `starting` has three values, not two: "true", "false",
and "unknown" for clubs that have not posted. Only the injured and confirmed
non-starters are dropped; "unknown" is kept, since filtering on "true" would
discard whole teams and bias toward early-posting clubs.

**Validating against Yahoo's own projections caught two real bugs**, which is
the entire argument for keeping that benchmark:

- `pitcher_points` had no win term at all, despite win being worth 4.
  Understated every starter by ~1.5 points; visible as a 2.6-point mean gap.
- `starter_bb` counts walks only, so the -0.9 hit-by-pitch penalty was missing.
  `starter_hbp` is now tracked separately rather than folded in, which would
  have changed what the prop output means by that name.

After both fixes: pitchers 13.28 -> 14.63 against Yahoo's 15.84, mean absolute
difference 2.75 -> 1.98, correlation 0.832. Hitters 6.55 against 7.06,
correlation 0.664.

**Two things unresolved.** A 1.21-point pitcher gap remains and is unexplained
-- possibly Yahoo being more optimistic on workload than our fitted hook, which
was itself verified against actual outs. And the hitter gaps are concentrated in
cheap players where we project 6-9 and Yahoo projects 2-3, which looks like
Yahoo discounting for partial playing time on exactly the "unknown"-lineup
players.

**The projected-lineup fallback captured nothing.** 24 players were priced from
projected lineups; none reached any of the five lineups, because averaging over
weighted candidates pulls them below confirmed starters. Defensible given a 5%
exact-repeat rate, but it should be a decision rather than a side effect --
especially since slots 1-4 hold their exact spot 65.4% of the time against 31.2%
for slots 6-9, so top-of-order players are far more predictable than the
aggregate suggests.

### Two DFS diagnoses overturned by measurement (2026-08-22)

Both of my explanations for the first build's oddities were wrong, and the
diagnostic that overturned them cost one run.

**"The projected-lineup fallback is too conservative."** It never executed.
`0 from projected lineups`, every priced hitter `source=posted`: StatsAPI had
lineups for all eleven games, so `resolve()` took the posted card every time.
Loosening the candidate weights would have been tuning a code path that did not
run. The fallback remains **untested in practice** and will only engage earlier
in the day, which is precisely when it matters.

**"Yahoo discounts cheap bats for playing time, and we should too."** Yahoo's
figures for that group are stale, not considered. Of 195 confirmed-starting
hitters, 20 carry a Yahoo projection under 3.0 and **every one is in the cheap
salary tercile** -- none in the mid or expensive tiers. Their own
Yahoo-published season average is 4.49, roughly double what Yahoo projects for
them tonight. Joc Pederson is batting *first* for Texas with a 2.2 projection.
Matching Yahoo here would make the model worse.

**What the investigation did find** is a real overstatement of the opposite
kind. The candidate accumulator divided by the weight of only those candidates
containing a player, which yields E[points | plays] -- so a fringe bat appearing
in one of three recent cards carried the same full-game projection as an
everyday regular. Now divided by total candidate weight, giving
E[points] = P(plays) x E[points | plays]. A player in every candidate is
unchanged (so top-of-order regulars are unaffected, as intended); an irregular
is discounted to what he is actually expected to produce.

Also applied: `earned_fraction` 0.92 on the pitcher run penalty, since the -2
applies to earned runs and the simulator tracks total runs.

Added `vs_own_avg`, the ratio of our projection to the player's own season
average, as a visible sanity band rather than an override -- a hitter promoted
to leadoff should beat his own average, but twice his established rate deserves
a look before it reaches a lineup.

### Yahoo board verified against the site (2026-08-22)

Cross-referenced our API pull against Yahoo's official rules page and its own
CSV export (`/v2/export/contestPlayers?contestId=`), which is what the site's
"Export players list" link serves.

- **Roster confirmed**: multi-game is `P P C 1B 2B 3B SS OF OF OF` -- two
  pitchers, ten players, 200 cap. Matches the configuration. Single-game
  contests use MEGASTAR/SUPERSTAR/FLEX and are not handled.
- **Batter scoring confirmed** exactly as recovered from the feed: 1B 2.6,
  2B 5.2, 3B 7.8, HR 10.4, R 1.9, RBI 1.9, BB 2.6, HBP 2.6, SB 4.2.
- **Pitcher scoring confirmed** exactly as supplied: W 4, K 2, ER -2, Out 1,
  BB/H/HBP -0.9.
- **Salaries and positions: 573 of 573 exact**, joining on name and team.
- **Single-position eligibility confirmed** in both the API and the export --
  no player carries more than one, and the export never uses a separator.

**A real bug surfaced from the two apparent mismatches.** Both were the same
collision: there are two Max Muncys, a Dodgers third baseman at 17 and an
Athletics third baseman at 11. Our Yahoo-name to MLBAM mapping keyed on
normalised name alone, so one overwrote the other and both Yahoo rows received
whichever id was written last -- one of them carrying the other's projection.
Tonight's second lineup rostered "Max Muncy ATH", so this was live. Now keyed on
(name, team) with a name-only fallback for everyone who is not half of a
duplicate pair.

Worth noting the verification was only possible because the export endpoint
exists; reading salaries off the rendered board would have been slower and less
complete.

### Saving fixed, and how it broke (2026-08-22)

`run_dfs.bat` produced nothing in `predictions/` because the build crashed at the
moment of saving: `NameError: name 'base' is not defined`.

Cause was a botched patch. The script applying it asserted on two anchors and the
second failed, so the file was never written; a follow-up edit then added only
the *second* half -- the `.txt` write and the confirmation print -- while the half
defining `base`, creating the directory and writing both CSVs was silently
missing. A syntax check passed, because the code was syntactically valid and only
referenced the undefined name on a path reached at the very end of a ten-minute
run. **A syntax check cannot substitute for executing the path that was
changed**, and this was found by the user running it rather than by me.

After the fix, `predictions/dfs_2026-08-22_1834.{txt,_lineups.csv,_pool.csv}`.

Both earlier fixes are visible in the output: lineup 1 now rosters **Max Muncy
LAD at 17** where the previous build had **Max Muncy ATH at 11**, confirming the
team-qualified id resolution changed which player the optimizer sees. Skubal
moved 22.0 -> 25.4 and lineup totals 109.0 -> 113.0 on the win term, HBP penalty
and earned-run factor.

`vs_own_avg` across the priced pool: hitters median 0.99, pitchers 1.04, with 7%
and 19% respectively above 1.5x. Exactly one rostered player trips the flag --
Cody Bradford at 1.68x (ours 19.7, Yahoo 16.7, his own average 11.8).

### Does recent form matter? Hitters, measured (2026-08-22)

Prompted by a DFS build rostering Joc Pederson in all five lineups while he was
hitting .067 over two weeks. Same test already run for pitchers, now for
hitters: 74,758 player-games with at least 40 prior games that season.

    correlation with next game's TB/PA
      season-to-date   0.0573
      last 12 games    0.0369      <- worse

    joint regression weights: season +0.3692, recent +0.0372
    R2: season alone 0.00329  + recent 0.00338  (gain +0.00009)

Recent form is a noisier estimate of the same quantity, exactly as with pitchers
(0.344 season-to-date against 0.296 for last-five-starts). The uniform
within-season weighting the model uses is correct.

**Concretely**, Pederson's recent rate sat 0.252 TB/PA below his season rate.
At the fitted recency weight of 0.037 that moves his projection about 0.009
TB/PA -- roughly 0.04 total bases, or **0.1 Yahoo points off 8.8**. He would
still appear in every lineup. There is not enough information in 30 plate
appearances to change the answer.

**The gap this does not close:** a slump caused by a hidden injury or a genuine
mechanical change *would* be predictive, and is indistinguishable from noise in
a TB/PA series. The defence there is Yahoo's DTD/IL flags and the active-roster
filter, not recency weighting.

Also noted: two players named in the review of that build -- "Edwardo Valencia"
and "Jamai Jones" -- do not appear in it. Valencia matches nothing in the
573-player pool; the closest to Jamai Jones is Jared Jones, who was priced but
not rostered. Whatever was being reviewed was a different build.

### Sanity band fixed: points per START, shrunk (2026-08-23)

`vs_own_avg` divided our full-game projection by Yahoo's `fantasyPointsPerGame`,
which averages over every appearance including one-PA cameos. For part-time
players that denominator is badly depressed -- Jahmai Jones started 28 of 70
games and averaged 1.4 PA in the other 42 -- so the flag fired hardest on
exactly the players it was least able to judge.

Now divided by points per game in games the player STARTED, computed from our
own PA table, shrunk toward the league starting average (6.01) with five starts
of prior. Median is unchanged (0.96 -> 0.95) while the share above 1.5 halves,
8% -> 4%: a bias removed, not a threshold moved.

The shrinkage was not cosmetic. A raw mean broke on thin history in the worst
possible way -- Corey Julks had one start and scored zero, giving a 0.00
denominator and a NaN ratio, which reads as *unflagged* for a player with no
starting record while he sat in a built lineup. Shrunk, he reads 1.17; Lowe with
78 starts moves only 7.55 -> 7.49.

    player            starts   per-start   ratio
    Corey Julks            1        5.46    1.17
    Jahmai Jones          28        4.73    1.65
    Eduardo Valencia      12        8.79    0.99
    Nathaniel Lowe        78        7.49    1.27

`own_start_avg` and `own_start_games` are written alongside so the denominator
and its sample size are visible, and Yahoo's figure is kept as `yahoo_fppg`.
Stolen bases (4.2) are omitted from the denominator -- the PA table lacks them
-- which nudges base-stealers' ratios slightly high.

**What this cannot do:** it compares a projection to the player's own history, so
it is blind to an error affecting everyone in the same direction. The missing
pitcher win term was invisible to it -- every pitcher was uniformly low and each
looked fine against himself -- and was caught only by the Yahoo diff. The two
checks are complementary and neither replaces the other. It is also deliberately
not wired into the optimizer: a promoted hitter *should* exceed his own rate, so
a flag is a prompt to look, never a filter.

### Lineup ceilings, and why a 107 projection is not "low" (2026-08-24)

Concern raised: builds project ~105 while contests are won with 150+. They are
different statistics. The optimizer maximises a *mean*; a winning score is the
extreme right tail across thousands of entries. A system whose mean matched
winning scores would be badly miscalibrated.

`lineup_distribution` answers it properly by drawing a shared simulation index
within each game and independent indices across games, so teammates stay
correlated -- when the leadoff man scores, the cleanup hitter's RBI lands in the
*same* simulated game. Summing independent per-player distributions would have
given a standard deviation nearer 18-20 and understated the tail by roughly
half. That covariance is the one structural advantage this simulator has over a
projection sheet.

    lineup   mean    sd    p90     p99   P(>150)
      1     107.5   26.1  141.4   172.2    5.6%
      2     103.5   25.5  136.2   166.2    3.8%
      3      99.7   27.0  134.7   166.3    3.5%

A 107-point lineup reaches 150 about one time in eighteen; five entries give
roughly a 20% chance at least one does. The projections were never low.

**Uniqueness.** `max_overlap` default 6 -> 1. Measured cost across 5 lineups,
no jitter: overlap 6 gives 21 of 50 distinct players at 534.8 total points;
overlap 1 gives 43 of 50 at 511.7; overlap 0 gives 50 of 50 at 491.1 (worst
lineup 107.3 -> 91.7). Live rebuild at overlap 1: max 1 shared between any two
lineups, 41 of 50 distinct.

**Not yet built: ceiling optimisation.** The objective is still expected points,
which is right for cash games and wrong for large-field tournaments. Maximising
P(lineup > threshold) would favour high-variance bats and naturally push toward
stacking correlated teammates. On this slate mean and ceiling happen to agree at
the top -- lineup 1 leads both -- so the change is not urgent, but lineup 3
carries the highest sd on the lowest mean, which is the shape a GPP build wants.

**Process note.** The ceiling report shipped broken: `samples` is keyed by MLBAM
id and the lookup passed names, so every call returned None and printed nothing.
Syntax passed. That is the third path this session that was written, checked for
syntax, and not executed -- after the `base` NameError in saving. Execute the
branch that changed.

### Tournament (ceiling) build, and two roster rules I did not know (2026-08-24)

For 360+ entry fields the mean is the wrong objective: roughly one lineup in
twenty clears 150, so about eighteen entries do, and winning means beating those
-- a target near a lineup's own 99th percentile.

P(total > threshold) is not linear in the selection, so the integer program
cannot maximise it. Two stages: generate many candidates against a linear proxy
(`mean + alpha*sd` per player), then score each candidate's true correlated
ceiling and keep the best subject to the overlap rule.

**Stacking had to be forced into generation, contrary to what I claimed.** I
argued stacks would emerge on their own because the scoring stage rewards
correlation, and that hard-coding them would assert what the simulator can
measure. Wrong: the generation objective is per-player and therefore
correlation-blind, so stacked lineups entered the candidate pool only by
accident -- the first ceiling build produced stacks of two to four across five
or six clubs, which is not a stack. Three quarters of candidates now get one
randomly chosen club's hitters boosted 35%, which only ensures stacked
candidates *exist* to be scored; the true correlated ceiling still decides.

**Two contest rules were being violated, both discovered by the user rather than
by me.** Yahoo allows at most 6 players from one club (pitchers included) and
requires at least 3 clubs represented. The stacked build was producing seven-
hitter stacks across two clubs, which would have been rejected at entry.

I had verified roster composition and scoring against Yahoo's rules page and
treated that as covering the constraints -- but those two answer what a valid
*roster* looks like, not what a valid *entry* looks like. Worse, I had removed
`max_per_team` in ceiling mode on the reasoning that it "suppressed the
correlation a tournament build wants", which treated a contest rule as a
tunable parameter.

The minimum-teams rule needed indicator variables, not another constraint row:
"at least three clubs" is a property of the *set* of teams used and cannot be
written as a bound on the player-slot variables. One binary per club, forced to
0 unless that club is represented, plus a lower bound on their sum. Both rules
were then tested against an objective deliberately built to violate them --
without the rule: 2 clubs (CWS 6, SD 4); with it: 3 clubs (CWS 6, SD 3, MIN 1).

Final legal build, contest 15744639:

    lineup   mean    sd    p90     p99   P(>150)   stack
      1     105.6   31.2  146.5   189.4    8.3%    CIN x6, 5 clubs
      2      98.7   29.3  136.7   178.8    4.9%    CHC x6, 4 clubs
      3      99.9   29.1  137.2   175.9    4.8%    ATH x6, 5 clubs

Against the mean-optimised build (sd 26.1, P(>150) 5.6%), the stacked legal
build reaches sd 31.2 and 8.3%. Cross-build comparison is not clean -- games
were locking as the evening progressed, so pool sizes differ (203 vs 138) --
but the standard deviation rising with stack depth is the correlation being
bought, and it is the whole point at this field size.

**Open:** other entry constraints may exist (minimum distinct games, multi-entry
limits). They should be asked for and encoded, not discovered one rejected
lineup at a time.

### Stack correlation with pitchers, measured (2026-08-24)

Question raised: when stacking a club's hitters, should the build also take that
club's starter to maximise correlation? Measured on 12 games x 4,000 sims with
a five-hitter stack (`stack_correlation.py`):

    correlation of a 5-hitter stack with
      its OWN club's starter     +0.032   (sd 0.027)
      the OPPOSING starter       -0.599   (sd 0.066)

    P(stack + that pitcher > 60 pts)
      own starter                 32.7%
      opposing starter            25.7%

**Own starter: negligible.** A pitcher earns from outs, strikeouts and runs not
allowed; his own offence produces none of those because he is not on the field
when they bat. The only channel is the win at +4, which is too narrow to build
a constraint around. The intuition is directionally right and practically empty.

**Opposing starter: the real effect, and negative.** Every run the stack scores
is a run he allowed at -2, and the damage pulls him early at 1 point per lost
out. So the useful rule is about what to avoid, not what to seek: never roster a
pitcher against your own hitters.

The build was violating it. Lineup 4 of `dfs_2026-08-24_1901` had Jeffrey
Springs (ATH) pitching against Minnesota while rostering two Minnesota hitters.
Now constrained: if a pitcher variable is 1, the sum of opposing hitters must be
0 (big-M on the hitter slot count). Opponent is derived from clubs sharing a
Yahoo game code, populated for 363 of 363 players.

Cost measured on the same board: 108.0 -> 107.1 projected, conflicts 2 -> 0.
Nine tenths of a mean point to remove a pairing that costs seven percentage
points of ceiling.

Note the optimizer was never blind to this -- it scores candidates on the true
joint distribution, so conflicted lineups were being penalised at selection.
What was missing was a hard rule, because the penalty is diluted across ten
players and can be outweighed by a strong pitcher's mean.

## Vegas totals and weather (2026-08-24)

**Market implied team totals beat the simulator, decisively.** Implied totals
come from the game total and run line: home = (total - runline)/2, away =
(total + runline)/2. On 214 team-games with both a recorded total and a result:

    simulator   RMSE 2.778   corr +0.182
    market      RMSE 2.591   corr +0.394
    best blend  w(market) = 0.9, RMSE 2.587

Blending improves on the market alone by 0.004 -- our team total adds almost
nothing once the market's is known. Same conclusion as the moneyline test. The
division of labour that follows: the market sets *how much* a club scores, the
simulator distributes it across players and preserves correlation.

Wired in at w=0.9. Points do **not** scale linearly with runs: fitted across
3,637 team-games, team hitter points go as runs^0.575 (eight runs yield 93
points against 41 at two). Total bases and walks accrue whether or not anyone
scores; only runs and RBI scale fully. A linear multiplier would have roughly
doubled the intended adjustment.

**Weather: real on its own, redundant once the market is known -- do not add
it.** Fetched 16,983 games of hourly history from Open-Meteo's archive (free, no
key, one call per park-season). Wind direction is stored as sin/cos rather than
converted to "blowing out", because that needs each park's centre-field bearing
which no API publishes; a per-park fit can recover the orientation from scoring.

With park fixed effects and no market term, temperature is worth +0.059 runs per
degree C -- a 10C swing moves a game 0.59 runs, 13% of a standard deviation.
Genuinely comparable to a moderate park factor.

But with the market total in the model, on 313 overlapping games:

    market alone            R2 0.08259
    + temperature           R2 0.08450  gain +0.0019   coef -0.0384
    + temperature and wind  R2 0.08821  gain +0.0056

**The temperature coefficient flips sign, +0.059 to -0.038.** The book already
prices weather and if anything over-adjusts. Applying our own positive
adjustment on top of a Vegas-anchored projection would push the wrong way. The
combined gain is also not significant: F ~ 0.95 on two parameters at n=313.

So weather arrives free through the Vegas anchor. The data is kept
(`game_weather.parquet`) because it would be needed if the market anchor were
ever dropped, and because the per-park wind fit is untested at this sample size.

Two fetch bugs worth remembering: the schedule endpoint returns no team
abbreviation without `hydrate=team` (every park keyed to "" and nothing
matched), and Open-Meteo's archive rejects a future end date, which silently
failed all thirty 2026 parks until the range was clamped to yesterday.
