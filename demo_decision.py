"""Worked example: the lefty-on-lefty call, and the pinch-hit counter.

Sets up a real 2025 game, freezes it in the seventh inning of a one-run game
with a runner on third and two outs, and prices the manager's options in win
probability.

    python demo_decision.py --sims 20000
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import bullpen
import config
import decisions as dec
import evaluate_sim
import outcome_model as om
import player_ratings as pr
import simulate as sim

SEASON = 2025


def build_context(pa, game, models, table, speed_ratings, rosters, hands,
                  reliever_vec, park_factors, calib, home_adv, lw):
    pen_arms, pen_hand = [], np.zeros((2, sim.MAX_PITCHERS), dtype=np.int8)
    pen_ids = []
    for ti, team in enumerate((game["away_team"], game["home_team"])):
        ids, w, avail, entry = rosters.as_of(team, game["game_date"],
                                             with_entry=True)
        seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
        sh = [hands.get(int(p), "R") for p in seq]
        pen_arms.append((seq, sh))
        pen_ids.append(list(zip(seq, sh)))
        for k, hd in enumerate(sh[:sim.MAX_PITCHERS - 1]):
            pen_hand[ti, k + 1] = 1 if hd == "L" else 0

    model = models[game["game_date"].month]
    cum = sim.build_cum_probs(
        model, game["away_lineup"], game["home_lineup"], game["away_stands"],
        game["home_stands"], game["away_starter"], game["home_starter"],
        game["away_hand"], game["home_hand"], reliever_vec,
        park_mult=park_factors.get(game["park"]), calibration=calib,
        home_advantage=home_adv, bullpen_arms=pen_arms)
    return cum, pen_hand, pen_ids, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pa = om.load_pa()
    prior = pa[pa["season"] < SEASON]
    oaa, speed = pr.load_ratings()
    speed_ratings = pr.SpeedRatings(speed, SEASON)

    print("preparing...")
    table = baserunning.build(prior, min_count=25, speed_ratings=speed_ratings)
    tables = om.build_count_tables(pa)
    om.TTO_MULT = om.fit_tto(prior)
    tto_mult = np.stack([om.TTO_MULT[t] for t in (1, 2, 3, 4)])
    park_factors = om.fit_park_factors(pa, SEASON)
    calib = om.fit_calibration(tables, pa, SEASON - 1,
                               park_factors=om.fit_park_factors(pa, SEASON - 1))
    home_adv = om.fit_home_advantage(pa, SEASON)
    reliever_vec = sim.league_reliever_vectors(prior)
    lw = baserunning.linear_weights(prior, baserunning.run_expectancy(prior))

    hook_model, _ = bullpen.fit_hook_model(
        pa, seasons=sorted(prior["season"].unique()))
    hook = sim.HazardHook(hook_model, bullpen.HOOK_FEATURES)
    rosters = bullpen.BullpenRosters(bullpen.relief_appearances(pa))
    hands = (pa[pa["is_pa"]].drop_duplicates("pitcher")
             .set_index("pitcher")["p_throws"].to_dict())

    games = evaluate_sim.extract_games(pa, SEASON)
    models = {}
    for month in sorted({g["game_date"].month for g in games}):
        done = pa[(pa["season"] == SEASON)
                  & (pd.to_datetime(pa["game_date"]).dt.month < month)]
        std = om.build_count_tables(done) if len(done) else None
        models[month] = om.OutcomeModel(tables, SEASON, season_to_date=std)

    # A game whose home bullpen carries both a left- and a right-hander.
    chosen = None
    for g in games:
        ids, w, avail, entry = rosters.as_of(g["home_team"], g["game_date"],
                                             with_entry=True)
        seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
        sh = [hands.get(int(p), "R") for p in seq]
        if "L" in sh and "R" in sh and len(seq) >= 4:
            # and an away lineup with a left-handed bat due up
            if "L" in list(g["away_stands"]):
                chosen = (g, seq, sh)
                break
    if chosen is None:
        raise SystemExit("no suitable game found")

    game, seq, sh = chosen
    cum, pen_hand, pen_ids, model = build_context(
        pa, game, models, table, speed_ratings, rosters, hands,
        reliever_vec, park_factors, calib, home_adv, lw)

    lineup_speed = np.array([
        [speed_ratings.get(int(p)) for p in game["away_lineup"]],
        [speed_ratings.get(int(p)) for p in game["home_lineup"]]],
        dtype=np.float32)
    lineup_side = np.array([
        [1 if s == "L" else 0 for s in game["away_stands"]],
        [1 if s == "L" else 0 for s in game["home_stands"]]], dtype=np.int8)

    kwargs = dict(tto_mult=tto_mult, lineup_speed=lineup_speed, hook=hook,
                  dispersion_sigma=0.19, pen_hand=pen_hand,
                  lineup_side=lineup_side)

    # The away side is batting; find a left-handed bat to put at the plate.
    lhb_slot = next(i for i, s in enumerate(game["away_stands"]) if s == "L")
    lhp_slot = 1 + sh.index("L")
    rhp_slot = 1 + sh.index("R")

    state = dec.GameState(
        inning=7, is_top=True, outs=2, base_code=4,      # runner on third
        score=(2, 3),                                    # home leads by one
        bat_idx=(lhb_slot, 0), pit_idx=(1, rhp_slot),
        bf=(0, 3), starter_bf=(24, 22), starter_pitches=(95.0, 88.0),
        starter_outs=(18.0, 17.0), starter_runs=(3.0, 2.0),
        base_speed=(0.0, 0.0, float(lineup_speed[0][max(lhb_slot - 1, 0)])))

    print(f"\ngame {game['game_pk']}: {game['away_team']} at {game['home_team']}"
          f", {pd.Timestamp(game['game_date']).date()}")
    print(f"situation: {state.describe()}")
    print(f"at the plate: away lineup slot {lhb_slot + 1} (bats L)")
    print(f"home bullpen order: {' '.join(sh)}")
    print(f"  right-hander in slot {rhp_slot}, left-hander in slot {lhp_slot}")

    lev = dec.leverage(cum, table, state, args.sims // 2, rng, **kwargs)
    print(f"\nleverage here: {lev * 100:.1f} points of win probability between "
          f"the run scoring and the batter being retired")

    print("\n--- DECISION 1: home manager, who faces the left-handed bat? ---")
    options = [
        dec.Option("stay with the right-hander", pit_idx=(1, rhp_slot)),
        dec.Option("bring in the left-hander", pit_idx=(1, lhp_slot)),
    ]
    results, best = dec.evaluate(cum, table, state, options, args.sims, rng,
                                 deciding_team=1, **kwargs)
    print(dec.format_results(results, best, deciding_team=1))

    print("\n--- DECISION 2: away manager's counter, if the lefty comes in ---")
    lhp_state = dec.GameState(**{**state.__dict__, "pit_idx": (1, lhp_slot),
                                 "bf": (0, 0)})
    # A right-handed bat off the bench: use the away lineup's best RHB as a
    # stand-in for a pinch hitter, which is what a bench bat approximates.
    rhb_slots = [i for i, s in enumerate(game["away_stands"]) if s == "R"]
    ph_vec = None
    if rhb_slots:
        best_rhb = rhb_slots[0]
        ph_vec = model.batter_vector(int(game["away_lineup"][best_rhb]), "R",
                                     "L")
    counter = [dec.Option("let the left-handed bat hit")]
    if ph_vec is not None:
        counter.append(dec.Option("pinch hit a right-handed bat",
                                  lineup_override=(0, lhb_slot, ph_vec)))
    cres, cbest = dec.evaluate(cum, table, lhp_state, counter, args.sims, rng,
                               deciding_team=0, **kwargs)
    print(dec.format_results(cres, cbest, deciding_team=0))

    print("\nnote: decision 1 is priced assuming the away manager does NOT "
          "counter.\nIf he does, the value of the pitching change is reduced by "
          "whatever\ndecision 2 recovers -- which is the point of reading them "
          "together.")


if __name__ == "__main__":
    main()
