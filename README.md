# MLB Game Simulator

Plate-appearance-level Monte Carlo simulation of MLB games, producing calibrated
*distributions* — win probability, run totals, team totals — rather than point
predictions, plus an in-game decision analyser that prices a manager's options
in win probability. The same engine prices player props and builds Yahoo daily
fantasy lineups.

## About this project

Built by Max Teply (Florida State, sport management with statistics and
economics minors) in summer 2026.

**How it was built.** The code was written with Claude Code, an AI coding
assistant. My part was the analysis: choosing the questions, deciding what to
test and how to hold data out, reading the results, and deciding what shipped
and what didn't. `FINDINGS.md` is the lab notebook for those decisions.

**Highlights**

- **Holdout 2025, 2,430 games:** the simulator blended with Elo reaches AUC
  0.581 and 56.3% accuracy. Always picking the home team gets 54.0%, and the
  closing betting line gets about 58.8%.
- **Bullpen and hook models were the largest single gain.** Bullpens record
  41.5% of all outs, not the third the first version assumed.
- **Half of all changes that improved their own metric made the whole
  simulator worse**, so every change is verified end to end before shipping.
- **Full 2026 season, nothing re-tuned:** run totals within 0.3% of actual
  (8.93 vs 8.96 per game) with the right spread, and 55.3% winners against
  52.8% for always picking the home team.
- **Honest null result against the market:** over 741 games with a recorded
  closing line, blending in any amount of the simulator makes the betting
  market's predictions worse. The line already contains everything the model
  knows about who wins. An earlier apparent gain from Elo turned out to be
  small-sample noise, caught by extending the blend sweep to weight 1.0.
- **A silent data gap, caught by a row count.** 217 September games were
  missing because partly downloaded data chunks were never refreshed. Nothing
  errored; a join that matched 525 games instead of 742 gave it away.
- **Eight silent bugs**, each producing plausible-looking output, are written
  up in `FINDINGS.md`. One produced an in-sample AUC of 0.996 and looked like a
  breakthrough.
- **Props and DFS.** Strikeout and hitter props, priced from the simulation, and
  an exact integer-programming lineup optimizer for Yahoo DFS. Benchmarking
  against Yahoo's own projections caught two scoring bugs on the first day.

**Data is not included** (about 300 MB of Statcast pitch data). The fetch scripts
rebuild it from public sources; see "Rebuilding from scratch" below.

---

Sibling project: [mlb-lineup-optimizer](https://github.com/mwteply64/mlb-lineup-optimizer)
(batter wOBA projection and lineup optimiser). This borrows its projection
*methodology* — per-outcome shrinkage, career/recent blending, the
evidence-weighted platoon prior — but not its data files, which hold only wOBA
point estimates and cannot be decomposed into the outcome vector a simulator
needs.

`FINDINGS.md` records the experiments behind every number here, including the
ones that failed. Several of those negative results are more useful than the
positive ones.

---

## Current results

Holdout 2025, 2,430 games. Shipped configuration: `--dispersion 0.15
--expand 1.2`, blended with Elo at w(sim) = 0.4.

| model | AUC | accuracy | log loss |
|---|---|---|---|
| always home | — | 54.03% | 0.68989 |
| simulator alone | 0.57324 | 54.32% | 0.68041 |
| Elo (record only) | 0.57706 | 56.42% | 0.67936 |
| **blend (shipped)** | **0.58068** | **56.30%** | **0.67729** |
| market, 2021 closing | — | 58.81% | 0.67040 |

Run distribution: mean 4.476 against an actual 4.445; team-runs standard
deviation ratio 0.9633.

**No shipped constant was selected on 2025.** An earlier version reported
0.58141 / 0.67700, with the dispersion sigma chosen by evaluating candidates on
the same season used to report the result. Re-selecting honestly cost 0.0007
AUC and 0.0003 nats — small, but the figures above no longer need an asterisk.

**The market is still ahead** — about 0.007 nats and 2.4 accuracy points, though
the market figure is from 2021 (the free archive stops there). The prospective
2026 comparison settled it: on 741 games with a recorded DraftKings closing
line, the market posted log loss 0.67175 against the simulator's 0.68249, and no
blend weight improved on the market alone. Details in `FINDINGS.md`.

### 2026 season

All 2,429 regular-season games, shipped configuration, nothing re-tuned:

| model | log loss | accuracy |
|---|---|---|
| always home | 0.69160 | 52.78% |
| **simulator** | **0.68647** | **55.29%** |

Runs per team-game 4.466 simulated against 4.480 actual; total-runs standard
deviation ratio 1.0082.

### The two constants behaved differently

**Expansion ×1.2 is solid.** Selected independently on 2024 and 2025, clean
interior optimum both times, symmetric on either side.

**Dispersion sigma is season-unstable.** 2024 wants 0.10, 2025 wants 0.19, and
no single value calibrates both: how much shared shock the simulation needs
depends on how variable that season's scoring actually was. The shipped 0.15 is
a disclosed compromise, not an optimum — it brackets the target, over-dispersing
2024 slightly (1.0287) and under-dispersing 2025 slightly (0.9633). A version
scaling with observed season variance would be more principled; two seasons is
not enough evidence to fit one.

### Known calibration gap: run level on 2024

Simulated runs per team-game come out **4.575 against an actual 4.392 on 2024**
(+4.2%), where 2025 is +0.7%. Almost certainly the league nowcast — the prior
comes from the two preceding seasons, and 2024 scored below them.

This is nearly invisible in win probability, because inflating both teams
symmetrically cancels. **It would bias totals directly**, and totals are the one
market where the simulator has no competitor, so it matters more than the win
metrics suggest. `LEAGUE_NOWCAST_PA` was never tuned.

## What accuracy to expect

Worth stating so results are read against the right yardstick:

- Straight-up winner: home-team-always is ~54%. Vegas closing lines ~58%.
  **A model at 58% has matched the state of the art**, not underperformed.
- Final score as a point estimate is not a meaningful target. Total runs has a
  standard deviation near 4.5; best achievable RMSE is about 4.0–4.2.
- Gate on **log loss, Brier, calibration, and AUC**. Accuracy discards nearly
  all the signal in a domain this noisy.

---

## Pipeline

| stage | script | what it does |
|---|---|---|
| 1 | `fetch_statcast.py` | resumable pitch-level download, half-month parquet chunks |
| 2 | `build_pa_events.py` | pitches → one row per plate appearance, five correctness gates |
| 3 | `outcome_model.py` | 8-way outcome vector for a batter–pitcher matchup |
| 4 | `baserunning.py` | empirical base-out transitions, run expectancy, linear weights |
| 5 | `player_ratings.py` | defence (OAA) and sprint speed, projected and regressed |
| 6 | `bullpen.py` | hook hazard model, roster-aware bullpens |
| 7 | `simulate.py` | vectorised Monte Carlo, arbitrary start state |
| 8 | `team_strength.py` | Elo ratings, team-strength expansion |

Evaluation: `evaluate_outcome_model.py`, `evaluate_sim.py`,
`baseline_models.py`, `calibrate_sim.py`, `tuner_outcome.py`.

Decisions: `decisions.py`, `decision_engine.py`, `leverage.py`, `positions.py`,
demos in `demo_decision.py` and `demo_engine.py`.

Odds: `fetch_odds.py` (historical), `record_odds.py` (prospective, scheduled
hourly as Windows task `MLB-Odds-Snapshot`).

### Rebuilding from scratch

```bash
python fetch_statcast.py && python build_pa_events.py && python fetch_player_ratings.py
```

```bash
python evaluate_sim.py --season 2025 --sims 1000 --dispersion 0.19 --expand 1.2
```

---

## Data

3,977,840 pitches → **1,034,138 completed plate appearances**, 13,744 games,
2021–2026, in about 60 MB of parquet.

All five build gates pass on the full span: **27,488 of 27,488 team-games
reconstruct their exact final score**, 100% of team-games use all nine lineup
slots, 100% of half-innings record exactly three outs.

Transition tables satisfy arithmetic identities — bases-loaded home run 4.00
runs, triple with two on 2.00, bases-loaded walk 1.00 — which makes them a
permanent gate rather than a one-time check.

**Season scope: 2021+.** The three-batter minimum (2020) ended the single-batter
lefty specialist, so earlier bullpen behaviour is a different game. Shift
restrictions and the pitch clock (2023) shift the batted-ball environment again,
so *tactical* models train on 2023+.

---

## Design decisions

**Predict after lineups post.** MLB StatsAPI publishes confirmed lineups 2–3
hours before first pitch. Making that the prediction point removes the need for
a projected-lineup model and keeps the backtest honest — backtesting on actual
lineups while deploying against projected ones measures a model you do not have.

**Entry state from the first pitch of a plate appearance, event from the last.**
Runners steal mid-plate-appearance; using the final pitch for both silently
credits stolen bases to the batter's outcome. Exit state is the next batter's
entry state, which is exact — baseball has no dead time between plate
appearances.

**Reached-on-error is defence, not offence.** Batter skill barely influences
whether a fielder muffs the ball, so ROE is classified `OUT` for the batter and
left to the fielding-team overlay.

**Descriptive for prediction, prescriptive for analysis.** The hook and bullpen
models reproduce what managers actually do, because that is what predicting a
game requires. The decision engine answers what they *should* do. Mixing the two
would model a manager who does not exist.

**Combine models at the output, never in the talent layer.** See `FINDINGS.md`;
this was measured, not assumed.

---

## In-game decision analysis

`decision_engine.py` enumerates every legal move in a state and ranks them in
one list: stand pat, pitching changes, pinch hits, two-step defensive
reshuffles, and pinch runners.

Each is priced through projected offence against the arm actually on the mound
(switch hitters taking the platoon advantage), defensive cost applied to the
remainder of the game, position eligibility as a hard legality filter, bench
depletion, and leverage for scale.

Worked example — HOU at LAA, top 8, one out, first and second, away down one,
leverage 2.31:

```
slot 4 (C) due up
  pinch hit 670623, 605170 slides to C   33.35%   +1.32  <-- best
  pinch hit 572233, 605170 slides to C   33.11%   +1.07
  pinch run 572233 at 1st                32.28%   +0.25
  stand pat                              32.03%
```

No bench player could catch, so the direct move is illegal; searching one
reshuffle deep finds a lineup player who can, freeing a bench first baseman.

**Common random numbers** halve the noise on comparisons: the standard deviation
of an estimated difference drops from 0.7906 to 0.3168 points, equivalent to six
times the simulations. This required making every random draw in `simulate()`
unconditional — two were conditional and desynchronised the branches.

Leverage (`leverage.py`) is available for any state and, unlike published
Leverage Index, accepts a matchup-specific outcome distribution — so who is
batting affects how much the moment is worth.

---

## Market baseline

**2021 closing moneylines, de-vigged: log loss 0.67040, Brier 0.23884, accuracy
58.81%**, mean vig 2.03%. Closing beats opening by 0.00084 nats.

The free archive stops at 2021 and states it will not be updated. Since 2021 is
the earliest season here, no model can yet be scored against the same games as
the benchmark.

**Prospective recording is live** (`record_odds.py`, hourly Windows task).
StatsAPI for the schedule and `gamePk`, ESPN's core API for DraftKings lines,
both keyless. Closing is reconstructed as the last snapshot before first pitch,
which is why hourly matters — a single daily snapshot landed a median of 19
hours early. A prospective comparison cannot be overfit, which makes it stronger
than any backtest.

Caveat: a single recreational book, vig 4.6–4.9% against the archive's 2.03%,
and recreational books shade toward public money. Treat results against it as an
upper bound on edge.

---

## Known gaps

- **No catcher framing.** Worth 1–2 wins a season for extreme catchers; acts on
  called strikes, so it belongs in the strikeout/walk channel. Savant's endpoint
  changed format.
- **No fielding arm.** OAA covers range; DRS adds arm strength and FanGraphs
  returns 403 via pybaseball. Manual CSV export is the path.
- **No stolen base model.** Statcast's `events` carries no baserunning events at
  all, and caught stealing cannot be reliably inferred from base-state changes.
  Needs Retrosheet. Magnitude is comparable to team defence, so low priority.
- **Team strength is under-dispersed**, corrected empirically (×1.2 expansion)
  rather than structurally. The principled fix is a two-level hierarchy on the
  *team* effect, and it should be paired with FanGraphs Depth Charts —
  forward-looking projections are the one team-strength input that could improve
  accuracy without converging on Elo.
- Reshuffle search is one move deep; three-player chains are not found.

---

## Validation discipline

Hold out full seasons. Gate on log loss, Brier, calibration and AUC against
three baselines at once: league constant, a team-strength model, and the market
**closing** line after vig. Component metrics are diagnostics, never gates —
this project has repeatedly produced clean component wins that lost at the game
level.

Verify every candidate end to end before shipping. Of the major changes tried,
roughly half improved their own metric and made the simulator worse.
