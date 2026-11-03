"""One evaluator: enumerate every legal move, price them all, rank them.

Previously the pieces sat separately — leverage in one module, projected hitting
in another, position eligibility in a third, defence in a fourth. This assembles
them, so a pinch hit is priced as what it actually is: an offensive upgrade, a
defensive downgrade, a burned bench player, and a position that still has to be
covered, all netted into a single win probability.

Three rules the engine enforces that a naive evaluator gets wrong:

* **A substitution is not free.** Taking a lineup slot means taking the glove
  that goes with it. Only 19 of 673 batters in 2025 had no fielding record, so
  designated hitter is very nearly the only costless slot to hit for.
* **Illegality is not expense.** If nobody available covers the vacated
  position, the move is off the board rather than merely bad.
* **Players do not come back.** Anyone already used is gone, which is what makes
  the bench a depleting resource and a decision in the sixth costly in the
  ninth.

Bench availability comes from the StatsAPI boxscore (exact, one call per game)
with an offline fallback inferred from recent appearances.
"""

import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config
import decisions as dec
import leverage as lev
import positions as pos_mod
import simulate as sim

# No custom User-Agent, matching the other fetchers. StatsAPI accepts any
# value; a malformed browser UA is what ESPN 403d on 2026-08-04.
UA = None
STATSAPI = "https://statsapi.mlb.com/api/v1"


# --------------------------------------------------------------------------
# Roster and bench
# --------------------------------------------------------------------------

def roster_from_boxscore(game_pk, timeout=30):
    """Exact rosters for one game: who started, who was available, who pitched.

    For a live game the `bench` and `bullpen` arrays are current availability.
    For a finished one they are whoever never got in, which combined with the
    play-by-play reconstructs availability at any point.
    """
    try:
        r = requests.get(f"{STATSAPI}/game/{game_pk}/boxscore",
                         timeout=timeout, headers=UA)
        if r.status_code != 200:
            return None
        box = r.json()
    except requests.RequestException:
        return None

    out = {}
    for side in ("away", "home"):
        t = box["teams"][side]
        players = t.get("players", {})
        pos_by_id = {}
        for key, p in players.items():
            pid = p.get("person", {}).get("id")
            if pid is None:
                continue
            codes = [x.get("code") for x in p.get("allPositions", []) or []]
            pos_by_id[int(pid)] = [int(c) for c in codes
                                   if c and str(c).isdigit() and 2 <= int(c) <= 9]
        out[side] = {
            "batting_order": [int(x) for x in t.get("battingOrder", [])],
            "batters": [int(x) for x in t.get("batters", [])],
            "bench": [int(x) for x in t.get("bench", [])],
            "bullpen": [int(x) for x in t.get("bullpen", [])],
            "pitchers": [int(x) for x in t.get("pitchers", [])],
            "positions": pos_by_id,
        }
    return out


def roster_from_appearances(pa, team, date, window_days=10):
    """Offline fallback: anyone who appeared for this team recently.

    Less exact than the boxscore -- it cannot tell an injured player from a
    rested one -- but it needs no network and is adequate for bulk analysis.
    """
    d = pa[pa["is_pa"]].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    date = pd.Timestamp(date)
    lo = date - pd.Timedelta(days=window_days)

    recent = d[(d["game_date"] < date) & (d["game_date"] >= lo)]
    bats = set(recent.loc[recent["batting_team"] == team, "batter"].astype(int))
    arms = set(recent.loc[recent["fielding_team"] == team, "pitcher"].astype(int))
    return {"batters": bats, "pitchers": arms}


def used_before(pa, game_pk, team, at_bat_number):
    """Everyone this team has already put in the game at that point.

    A player who has appeared and been replaced cannot return, so this set is
    what makes the bench deplete over nine innings.
    """
    d = pa[(pa["game_pk"] == game_pk) & pa["is_pa"]
           & (pa["at_bat_number"] < at_bat_number)]
    used = set(d.loc[d["batting_team"] == team, "batter"].astype(int))
    used |= set(d.loc[d["fielding_team"] == team, "pitcher"].astype(int))
    for i in range(2, 10):
        col = f"fielder_{i}"
        if col in d.columns:
            used |= set(d.loc[d["fielding_team"] == team, col].dropna().astype(int))
    return used


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

@dataclass
class Move:
    """One legal thing a manager can do here."""
    name: str
    kind: str          # stand_pat | pitching_change | pinch_hit | pinch_run
    pit_idx: tuple = None
    batter_slot: int = None
    batter_id: int = None
    batter_team: int = None
    defense_delta: float = 0.0      # runs per defensive PA, signed for the team
    vacated_position: int = None
    # Pinch running: which base, and how fast the new runner is.
    base: int = None
    runner_speed: float = None
    # Multi-step reshuffle: [(position, player_id_now_there), ...]. Used when
    # nobody on the bench covers the vacated position directly but a fielder
    # already in the game does, and the bench player covers *his* spot.
    reshuffle: tuple = ()
    note: str = ""


@dataclass
class Context:
    """Static per-game inputs plus the fitted models."""
    model: object
    table: object
    lw: np.ndarray
    cum_base: np.ndarray
    away_lineup: np.ndarray
    home_lineup: np.ndarray
    away_stands: np.ndarray
    home_stands: np.ndarray
    lineup_speed: np.ndarray
    lineup_side: np.ndarray
    pen_hand: np.ndarray
    defense: object = None
    positions: object = None
    slot_positions: dict = field(default_factory=dict)   # team -> {slot: pos}
    defense_h: list = None
    sim_kwargs: dict = field(default_factory=dict)
    stands: dict = field(default_factory=dict)   # player_id -> "L" / "R"
    speed: object = None                          # SpeedRatings, for runners

    def bat_side(self, player_id, vs_hand):
        """Which side a hitter stands on against this pitcher.

        Switch hitters take the platoon advantage, which is what makes them
        immune to the lefty-on-lefty move and therefore worth distinguishing.
        """
        s = self.stands.get(int(player_id), "R")
        if s == "S":
            return "R" if vs_hand == "L" else "L"
        return s


def enumerate_moves(ctx, state, team, bench_ids, bullpen_slots=(),
                    max_hitters=6):
    """Every legal move for `team` in this state.

    Pinch hitters are filtered on position eligibility, not just offence: a bat
    that cannot cover the vacated slot is excluded outright rather than ranked
    badly.
    """
    moves = [Move("stand pat", "stand_pat")]

    for slot in bullpen_slots:
        moves.append(Move(f"bring in bullpen arm {slot}", "pitching_change",
                          pit_idx=(state.pit_idx[0] if team == 1 else slot,
                                   slot if team == 1 else state.pit_idx[1])))

    slot = state.bat_idx[team]
    vacated = ctx.slot_positions.get(team, {}).get(slot)
    lineup = ctx.home_lineup if team == 1 else ctx.away_lineup
    outgoing = int(lineup[slot])

    # Whichever arm is on the mound for the other side right now.
    opp_slot = state.pit_idx[1 - team]
    vs_hand = "L" if ctx.pen_hand[1 - team][opp_slot] else "R"

    scored = []
    for pid in bench_ids:
        if ctx.positions is not None and vacated is not None:
            if not ctx.positions.can_play(pid, vacated):
                continue
        d_delta = 0.0
        if ctx.defense is not None and vacated is not None:
            d_delta = ctx.defense.get(pid) - ctx.defense.get(outgoing)
        side = ctx.bat_side(pid, vs_hand)
        # Rank candidates by projected offence against the arm actually facing
        # them, so the shortlist is the bats worth considering rather than the
        # first few on the roster.
        val = float(np.dot(ctx.model.batter_vector(int(pid), side, vs_hand),
                           ctx.lw))
        scored.append((val, pid, d_delta, side))

    scored.sort(key=lambda x: -x[0])
    for _, pid, d_delta, side in scored[:max_hitters]:
        label = f"pinch hit {pid}"
        note = ("no defensive cost (designated hitter slot)" if vacated is None
                else f"covers {pos_mod.POS_NAMES.get(vacated, vacated)}, "
                     f"defence {d_delta * 6100:+.1f} runs per 162")
        moves.append(Move(label, "pinch_hit", batter_slot=slot, batter_id=pid,
                          batter_team=team, defense_delta=d_delta,
                          vacated_position=vacated, note=note))

    moves += _reshuffle_moves(ctx, state, team, bench_ids, slot, vacated,
                              outgoing, lineup)
    moves += _pinch_run_moves(ctx, state, team, bench_ids, lineup)
    return moves


def _reshuffle_moves(ctx, state, team, bench_ids, slot, vacated, outgoing,
                     lineup, max_moves=3):
    """Two-step substitutions: a fielder slides over, the bench covers him.

    Only generated when the vacated position has no direct replacement --
    otherwise the simple move dominates and this just clutters the board. This
    is what makes a catcher's spot genuinely blocked rather than merely
    awkward: sliding someone to catch requires a player who has actually caught.
    """
    if vacated is None or ctx.positions is None:
        return []
    if any(ctx.positions.can_play(p, vacated) for p in bench_ids):
        return []

    out = []
    slot_pos = ctx.slot_positions.get(team, {})
    for other_slot, other_pos in slot_pos.items():
        if other_slot == slot:
            continue
        mover = int(lineup[other_slot])
        if not ctx.positions.can_play(mover, vacated):
            continue
        for pid in bench_ids:
            if not ctx.positions.can_play(pid, other_pos):
                continue
            # The bench bat enters at the vacated slot; the mover changes
            # position but keeps his place in the order.
            d = 0.0
            if ctx.defense is not None:
                d = (ctx.defense.get(mover) - ctx.defense.get(outgoing)
                     + ctx.defense.get(pid) - ctx.defense.get(mover))
            out.append(Move(
                f"pinch hit {pid}, {mover} slides to "
                f"{pos_mod.POS_NAMES.get(vacated, vacated)}",
                "pinch_hit", batter_slot=slot, batter_id=pid,
                batter_team=team, defense_delta=d, vacated_position=vacated,
                reshuffle=((vacated, mover), (other_pos, pid)),
                note=(f"two-step: {mover} covers "
                      f"{pos_mod.POS_NAMES.get(vacated, vacated)}, {pid} takes "
                      f"{pos_mod.POS_NAMES.get(other_pos, other_pos)}")))
            if len(out) >= max_moves:
                return out
    return out


def _pinch_run_moves(ctx, state, team, bench_ids, lineup, max_moves=3):
    """Replace a baserunner with someone faster.

    The gain runs through the speed-conditioned transition tables directly: a
    fast runner on second scores from second far more often. The cost is that
    the runner's place in the order goes to the pinch runner, who is usually a
    worse hitter, and his glove has to be covered too.
    """
    if ctx.positions is None:
        return []

    out = []
    slot_pos = ctx.slot_positions.get(team, {})
    for base in range(3):
        if not ((state.base_code >> base) & 1):
            continue
        runner_slot = state.base_slots[base]
        if runner_slot is None or runner_slot < 0:
            continue

        current_speed = (state.base_speed[base] if state.base_speed
                         else ctx.lineup_speed[team][runner_slot])
        outgoing = int(lineup[runner_slot])
        vacated = slot_pos.get(runner_slot)

        cands = []
        for pid in bench_ids:
            if vacated is not None and not ctx.positions.can_play(pid, vacated):
                continue
            spd = ctx.speed.get(pid) if ctx.speed else None
            if spd is None or spd <= current_speed + 0.3:
                continue        # not enough faster to be worth a roster spot
            cands.append((spd, pid))

        cands.sort(reverse=True)
        for spd, pid in cands[:max_moves]:
            d = 0.0
            if ctx.defense is not None and vacated is not None:
                d = ctx.defense.get(pid) - ctx.defense.get(outgoing)
            out.append(Move(
                f"pinch run {pid} at {['1st', '2nd', '3rd'][base]}",
                "pinch_run", base=base, runner_speed=float(spd),
                batter_slot=runner_slot, batter_id=pid, batter_team=team,
                defense_delta=d, vacated_position=vacated,
                note=(f"{spd:.1f} ft/s vs {current_speed:.1f}; takes lineup "
                      f"slot {runner_slot + 1} for the rest of the game")))
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def apply_move(ctx, state, move):
    """Return (cum_probs, state) reflecting the move having been made."""
    cum = ctx.cum_base
    st = state

    if move.kind == "pinch_hit":
        cum = ctx.cum_base.copy()
        team = move.batter_team
        opp_hand = "L" if ctx.pen_hand[1 - team][state.pit_idx[1 - team]] else "R"
        side = ctx.bat_side(move.batter_id, opp_hand)
        vec = ctx.model.batter_vector(int(move.batter_id), side, opp_hand)

        # Defence changes for the REST of the game, so it belongs in the
        # opposing team's offensive vectors, not in a post-hoc adjustment.
        for pslot in range(cum.shape[1]):
            cum[team, pslot, move.batter_slot] = np.cumsum(vec / vec.sum())

        if move.defense_delta and ctx.defense_h is not None:
            cum = _apply_defense_change(ctx, cum, team, move.defense_delta)

    elif move.kind == "pinch_run":
        cum = ctx.cum_base.copy()
        team = move.batter_team
        opp_slot = state.pit_idx[1 - team]
        opp_hand = "L" if ctx.pen_hand[1 - team][opp_slot] else "R"
        side = ctx.bat_side(move.batter_id, opp_hand)
        vec = ctx.model.batter_vector(int(move.batter_id), side, opp_hand)

        # The runner's place in the order goes with him for the rest of the
        # game -- that is the whole cost of the move.
        for pslot in range(cum.shape[1]):
            cum[team, pslot, move.batter_slot] = np.cumsum(vec / vec.sum())

        speeds = list(state.base_speed) if state.base_speed else [0.0, 0.0, 0.0]
        speeds[move.base] = move.runner_speed
        st = dec.GameState(**{**state.__dict__, "base_speed": tuple(speeds)})

        if move.defense_delta and ctx.defense_h is not None:
            cum = _apply_defense_change(ctx, cum, team, move.defense_delta)

    elif move.kind == "pitching_change":
        st = dec.GameState(**{**state.__dict__, "pit_idx": move.pit_idx,
                              "bf": (0, 0)})

    return cum, st


def _apply_defense_change(ctx, cum, team, delta_runs):
    """Fold a fielding change into the OPPOSING side's offensive vectors.

    A team's defence acts on the other team's hitting, so that is where the
    adjustment belongs. Applied for the remainder of the game rather than as a
    separate line item, which is what lets one number carry the whole trade.
    """
    import player_ratings as pr
    h = pr.defense_hit_suppression(-delta_runs, ctx.model.league, ctx.lw)
    opp = 1 - team
    for pslot in range(cum.shape[1]):
        for s in range(9):
            v = np.diff(np.concatenate([[0.0], cum[opp, pslot, s]]))
            v = pr.apply_defense(v, h)
            cum[opp, pslot, s] = np.cumsum(v)
    return cum


def evaluate_moves(ctx, state, moves, team, n_sims, rng, common_random=True):
    """Win probability for `team` under each move.

    With `common_random`, every branch is simulated on the same stream of
    random numbers. The branches then share their luck -- the same sequence of
    hits and outs, the same hook decisions -- so differencing them cancels most
    of the noise and leaves the effect of the decision itself. What matters
    here is the *difference* between options, not any single option's absolute
    win probability, and that difference is exactly what this sharpens.

    The branches do drift apart once an early outcome differs, so the
    correlation decays through the game rather than holding perfectly. It is
    still worth several times the simulation count.
    """
    seed = int(rng.integers(2 ** 31 - 1))
    rows = []
    for mv in moves:
        cum, st = apply_move(ctx, state, mv)
        branch_rng = (np.random.default_rng(seed) if common_random else rng)
        p_home = dec.win_probability(cum, ctx.table, st, n_sims, branch_rng,
                                     **ctx.sim_kwargs)
        p = p_home if team == 1 else 1.0 - p_home
        rows.append({"move": mv, "win_prob": p})

    base = next(r["win_prob"] for r in rows if r["move"].kind == "stand_pat")
    for r in rows:
        r["delta"] = r["win_prob"] - base
    rows.sort(key=lambda r: -r["win_prob"])
    return rows


def format_moves(rows, team, leverage_value=None):
    lines = []
    who = "home" if team == 1 else "away"
    head = f"  {'move':<30} {who + ' win %':>11} {'vs stand pat':>13}"
    if leverage_value is not None:
        head += f"    (leverage {leverage_value:.2f})"
    lines.append(head)
    for r in rows:
        mv = r["move"]
        star = "  <-- best" if r is rows[0] else ""
        lines.append(f"  {mv.name:<30} {r['win_prob'] * 100:>10.2f}% "
                     f"{r['delta'] * 100:>+12.2f}{star}")
        if mv.note:
            lines.append(f"      {mv.note}")
    return "\n".join(lines)
