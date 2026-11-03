"""End-to-end decision analysis on a real game state.

Everything netted into one number: offensive change from the pinch hitter, the
defensive cost of the glove he replaces, whether he can legally cover the
position at all, who is left on the bench afterwards, and what it is all worth
given the leverage of the moment.

    python demo_engine.py --sims 15000
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import bullpen
import decision_engine as eng
import decisions as dec
import evaluate_sim
import leverage as lev
import outcome_model as om
import player_ratings as pr
import positions as pos_mod
import simulate as sim

SEASON = 2025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pa = om.load_pa()
    prior = pa[pa["season"] < SEASON]
    oaa, speed = pr.load_ratings()
    speed_ratings = pr.SpeedRatings(speed, SEASON)
    defense = pr.DefenseRatings(oaa, pr.defensive_exposure(pa), SEASON)

    print("preparing models...")
    table = baserunning.build(prior, min_count=25, speed_ratings=speed_ratings)
    tables = om.build_count_tables(pa)
    om.TTO_MULT = om.fit_tto(prior)
    tto_mult = np.stack([om.TTO_MULT[t] for t in (1, 2, 3, 4)])
    reliever_vec = sim.league_reliever_vectors(prior)
    lw = baserunning.linear_weights(prior, baserunning.run_expectancy(prior))
    model = om.OutcomeModel(tables, SEASON)
    hook_model, _ = bullpen.fit_hook_model(
        pa, seasons=sorted(prior["season"].unique()))
    hook = sim.HazardHook(hook_model, bullpen.HOOK_FEATURES)
    rosters = bullpen.BullpenRosters(bullpen.relief_appearances(pa))
    hands = (pa[pa["is_pa"]].drop_duplicates("pitcher")
             .set_index("pitcher")["p_throws"].to_dict())
    stands = (pa[pa["is_pa"]].drop_duplicates("batter")
              .set_index("batter")["stand"].to_dict())
    wp = lev.load_win_probability()
    elig = pos_mod.load_eligibility()
    positions = pos_mod.Positions(elig, SEASON)

    games = evaluate_sim.extract_games(pa, SEASON)
    game = None
    for g in games:
        box = eng.roster_from_boxscore(g["game_pk"])
        if box and len(box["away"]["bench"]) >= 3:
            game, boxscore = g, box
            break
    if game is None:
        raise SystemExit("no game with a usable bench found")

    pen_arms, pen_hand = [], np.zeros((2, sim.MAX_PITCHERS), dtype=np.int8)
    for ti, team in enumerate((game["away_team"], game["home_team"])):
        ids, w, avail, entry = rosters.as_of(team, game["game_date"],
                                             with_entry=True)
        seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
        sh = [hands.get(int(p), "R") for p in seq]
        pen_arms.append((seq, sh))
        for k, hd in enumerate(sh[:sim.MAX_PITCHERS - 1]):
            pen_hand[ti, k + 1] = 1 if hd == "L" else 0

    cum_base = sim.build_cum_probs(
        model, game["away_lineup"], game["home_lineup"], game["away_stands"],
        game["home_stands"], game["away_starter"], game["home_starter"],
        game["away_hand"], game["home_hand"], reliever_vec,
        bullpen_arms=pen_arms)

    lineup_speed = np.array(
        [[speed_ratings.get(int(p)) for p in game["away_lineup"]],
         [speed_ratings.get(int(p)) for p in game["home_lineup"]]],
        dtype=np.float32)
    lineup_side = np.array(
        [[1 if s == "L" else 0 for s in game["away_stands"]],
         [1 if s == "L" else 0 for s in game["home_stands"]]], dtype=np.int8)

    slot_positions = {}
    for ti, team in enumerate((game["away_team"], game["home_team"])):
        lp = pos_mod.lineup_positions(pa, game["game_pk"], team)
        slot_positions[ti] = {v["slot"] - 1: p for p, v in lp.items()}

    ctx = eng.Context(
        model=model, table=table, lw=lw, cum_base=cum_base,
        away_lineup=game["away_lineup"], home_lineup=game["home_lineup"],
        away_stands=game["away_stands"], home_stands=game["home_stands"],
        lineup_speed=lineup_speed, lineup_side=lineup_side, pen_hand=pen_hand,
        defense=defense, positions=positions, slot_positions=slot_positions,
        defense_h=[0.0, 0.0], stands=stands, speed=speed_ratings,
        sim_kwargs=dict(tto_mult=tto_mult, lineup_speed=lineup_speed,
                        hook=hook, dispersion_sigma=0.19, pen_hand=pen_hand,
                        lineup_side=lineup_side))

    lhp_slots = [i for i in range(1, sim.MAX_PITCHERS) if pen_hand[1][i]]
    lhp_slot = lhp_slots[0] if lhp_slots else 1

    slot = 3
    # Runners on first and second belonging to the two slots ahead of the bat.
    r1_slot, r2_slot = (slot - 1) % 9, (slot - 2) % 9
    state = dec.GameState(
        inning=8, is_top=True, outs=1, base_code=3,       # first and second
        score=(2, 3), bat_idx=(slot, 0), pit_idx=(1, lhp_slot), bf=(0, 1),
        starter_bf=(24, 22), starter_pitches=(96.0, 90.0),
        starter_outs=(18.0, 18.0), starter_runs=(3.0, 2.0),
        base_speed=(float(lineup_speed[0][r1_slot]),
                    float(lineup_speed[0][r2_slot]), 0.0),
        base_slots=(r1_slot, r2_slot, -1))

    margin = state.score[0] - state.score[1]
    li = lev.leverage(wp, table, state.inning, state.is_top, state.outs,
                      state.base_code, margin)
    ref = lev.leverage(wp, table, 5, True, 1, 0, 0)

    print(f"\ngame {game['game_pk']}: {game['away_team']} at "
          f"{game['home_team']}, {pd.Timestamp(game['game_date']).date()}")
    print(f"situation: {state.describe()}")
    print(f"leverage: {li / ref:.2f}")

    used = eng.used_before(pa, game["game_pk"], game["away_team"],
                           at_bat_number=10_000)
    bench = [p for p in boxscore["away"]["bench"] if p not in
             set(int(x) for x in game["away_lineup"])]
    print(f"\naway bench from the boxscore: {len(boxscore['away']['bench'])} "
          f"players, {len(bench)} not already in the lineup")

    print("\nposition coverage, slot by slot:")
    playable = []
    for s in range(9):
        vac = slot_positions[0].get(s)
        vname = pos_mod.POS_NAMES.get(vac, "DH")
        elig_n = sum(1 for p in bench
                     if vac is None or positions.can_play(p, vac))
        print(f"  slot {s + 1} ({vname:<3}): {elig_n} of {len(bench)} "
              f"bench players can cover it")
        if elig_n >= 2:
            playable.append(s)

    for p in bench:
        print(f"  bench {p}: {positions.describe(p)[:64]}")

    # The catcher case is the constraint working; a covered slot shows the
    # trade-off the engine exists to price.
    for s in [slot] + playable[:1]:
        vac = slot_positions[0].get(s)
        vname = pos_mod.POS_NAMES.get(vac, "DH")
        st = dec.GameState(**{**state.__dict__, "bat_idx": (s, 0)})
        moves = eng.enumerate_moves(ctx, st, team=0, bench_ids=bench)
        print(f"\n--- slot {s + 1} ({vname}) due up, incumbent "
              f"{int(game['away_lineup'][s])} ---")
        if len(moves) == 1:
            print("  no legal substitution: nobody on the bench covers this "
                  "position.\n  A naive evaluator would have recommended one "
                  "anyway.")
            continue
        rows = eng.evaluate_moves(ctx, st, moves, team=0, n_sims=args.sims,
                                  rng=rng)
        print(eng.format_moves(rows, team=0, leverage_value=li / ref))


if __name__ == "__main__":
    main()
