"""In-game decision analysis: what a manager's choice is actually worth.

Freeze a game state, branch on the decision, run each branch forward many
times, and compare win probability. The difference between branches is the
decision's value in win probability -- the only unit that matters, because it
already accounts for inning, score, and what is left in both bullpens.

This is the *prescriptive* side. The simulator's hook and bullpen models are
deliberately **descriptive**: they reproduce what managers actually do, because
that is what predicting a game requires. Asking what a manager *should* do is a
different question, and it is the one this module answers.

Two properties make the answers trustworthy in a way a run-expectancy table is
not:

* **It plays the game out.** A run expectancy matrix says what a base-out state
  is worth in runs. In the seventh inning of a one-run game, runs and wins are
  not the same currency -- the fourth run of an inning barely matters and the
  tying run matters enormously. Simulating to the end prices that correctly.
* **It knows who is left.** The value of using your best reliever now depends
  on who pitches the ninth if you do. Because the branch runs forward through
  the actual bullpen sequence, that cost is counted rather than assumed away.

Counter-moves are handled as a shallow game tree. Depth two is enough: real
managers do not search deeper, and the branching cost compounds fast.
"""

import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

import numpy as np

import config
import simulate as sim

OUTCOMES = config.OUTCOMES


@dataclass
class GameState:
    """A frozen situation, ready to be played forward.

    Scores and indices are per team with 0 = away and 1 = home, matching the
    simulator throughout.
    """
    inning: int = 7
    is_top: bool = False
    outs: int = 2
    base_code: int = 4          # 1B=1, 2B=2, 3B=4
    score: tuple = (3, 2)       # (away, home)
    bat_idx: tuple = (0, 4)     # next batter's lineup slot, per team
    pit_idx: tuple = (1, 1)     # current pitcher slot, per team
    bf: tuple = (0, 0)          # batters faced by the current pitcher
    starter_bf: tuple = (22, 22)
    starter_pitches: tuple = (88.0, 88.0)
    starter_outs: tuple = (17.0, 17.0)
    starter_runs: tuple = (2.0, 2.0)
    base_speed: tuple = None    # speed of the runner on each base
    # Which lineup slot each baserunner belongs to (-1 = base empty). A pinch
    # runner takes over that slot in the order, so without this the cost of the
    # move -- losing that bat for the rest of the game -- cannot be priced.
    base_slots: tuple = (-1, -1, -1)

    def to_dict(self, n_sims, speed_default=27.2):
        bs = self.base_speed
        if bs is None:
            bs = [speed_default if (self.base_code >> k) & 1 else 0.0
                  for k in range(3)]
        return {
            "inning": self.inning, "is_top": self.is_top, "outs": self.outs,
            "base_code": self.base_code,
            "score": np.tile(np.array(self.score, dtype=np.int16), (n_sims, 1)),
            "bat_idx": np.tile(np.array(self.bat_idx, dtype=np.int8), (n_sims, 1)),
            "pit_idx": np.tile(np.array(self.pit_idx, dtype=np.int8), (n_sims, 1)),
            "bf": np.tile(np.array(self.bf, dtype=np.int16), (n_sims, 1)),
            "starter_bf": np.tile(np.array(self.starter_bf, dtype=np.int16), (n_sims, 1)),
            "starter_pitches": np.tile(np.array(self.starter_pitches, dtype=np.float32), (n_sims, 1)),
            "starter_outs": np.tile(np.array(self.starter_outs, dtype=np.float32), (n_sims, 1)),
            "starter_runs": np.tile(np.array(self.starter_runs, dtype=np.float32), (n_sims, 1)),
            "base_speed": np.tile(np.array(bs, dtype=np.float32), (n_sims, 1)),
        }

    def describe(self):
        bases = []
        for k, name in enumerate(("1st", "2nd", "3rd")):
            if (self.base_code >> k) & 1:
                bases.append(name)
        where = ", ".join(bases) if bases else "bases empty"
        half = "top" if self.is_top else "bottom"
        lead = self.score[1] - self.score[0]
        margin = ("tied" if lead == 0 else
                  f"home {'up' if lead > 0 else 'down'} {abs(lead)}")
        return (f"{half} {self.inning}, {self.outs} out, {where}, "
                f"{margin} ({self.score[0]}-{self.score[1]})")


def win_probability(cum_probs, table, state, n_sims, rng, **kwargs):
    """Home win probability from a frozen state, ties split."""
    away, home, _ = sim.simulate(cum_probs, table, n_sims, rng,
                                 initial_state=state.to_dict(n_sims), **kwargs)
    return float((home > away).mean() + 0.5 * (home == away).mean())


@dataclass
class Option:
    """One branch of a decision."""
    name: str
    pit_idx: tuple = None       # override the pitcher slot per team
    lineup_override: tuple = None   # (team, slot, outcome vector) pinch hitter
    note: str = ""


def evaluate(cum_probs, table, state, options, n_sims, rng, deciding_team=1,
             **kwargs):
    """Win probability under each option, from the deciding team's side.

    `deciding_team` is whose decision it is (0 away, 1 home); results are
    reported as that team's win probability so higher is always better for the
    manager making the call.
    """
    results = []
    for opt in options:
        cum = cum_probs
        if opt.lineup_override is not None:
            team, slot, vec = opt.lineup_override
            cum = cum_probs.copy()
            # Same batter against every pitcher the opposing side may use.
            for pslot in range(cum.shape[1]):
                cum[team, pslot, slot] = np.cumsum(vec / vec.sum())

        st = state
        if opt.pit_idx is not None:
            st = GameState(**{**state.__dict__, "pit_idx": opt.pit_idx,
                              "bf": (0, 0)})

        p_home = win_probability(cum, table, st, n_sims, rng, **kwargs)
        p = p_home if deciding_team == 1 else 1.0 - p_home
        results.append((opt, p))

    best = max(results, key=lambda r: r[1])
    return results, best


def leverage(cum_probs, table, state, n_sims, rng, **kwargs):
    """How much this situation can swing, in win probability.

    The gap between the batting team scoring on this plate appearance and being
    retired. High leverage is where decisions are worth making; in a blowout
    every option is worth the same and the manager's choice does not matter.
    """
    scored = GameState(**{**state.__dict__,
                          "score": (state.score[0] + (1 if state.is_top else 0),
                                    state.score[1] + (0 if state.is_top else 1))})
    retired = GameState(**{**state.__dict__,
                           "outs": min(state.outs + 1, 2),
                           "base_code": state.base_code})
    p_scored = win_probability(cum_probs, table, scored, n_sims, rng, **kwargs)
    p_retired = win_probability(cum_probs, table, retired, n_sims, rng, **kwargs)
    return abs(p_scored - p_retired)


def format_results(results, best, deciding_team=1):
    lines = []
    team = "home" if deciding_team == 1 else "away"
    lines.append(f"  {'option':<34} {team + ' win %':>12}   {'vs best':>8}")
    for opt, p in sorted(results, key=lambda r: -r[1]):
        gap = p - best[1]
        mark = "  <-- best" if opt is best[0] else ""
        lines.append(f"  {opt.name:<34} {p * 100:>11.2f}% {gap * 100:>+8.2f}"
                     f"{mark}")
        if opt.note:
            lines.append(f"      {opt.note}")
    return "\n".join(lines)
