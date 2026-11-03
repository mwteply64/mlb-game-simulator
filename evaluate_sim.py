"""Validate the game simulator against real seasons.

This is the step-two checkpoint. Two questions matter, in this order:

1. **Does the run distribution match?** Not just the mean -- the *spread*. A
   Markov simulation with independent plate appearances systematically
   under-disperses team run totals, because real games have shared shocks
   (weather, umpire zone, a staff-wide bad night) that independent sampling
   cannot produce. If the simulated standard deviation is short, totals markets
   are untrustworthy no matter how good the mean looks.

2. **Is win probability calibrated?** Scored against the same three baselines
   the rest of the project uses.

Games are simulated with actual lineups and actual starting pitchers, which is
the information available at the stated prediction point (after lineups post).

    python evaluate_sim.py --season 2025 --sims 500 --games 800
"""

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import bullpen
import config
import outcome_model as om
import player_ratings as pr
import simulate as sim
import team_strength as ts

OUTCOMES = config.OUTCOMES


def extract_games(pa, season):
    """Actual lineups, starters, and results for every game in a season."""
    d = pa[(pa["season"] == season) & pa["is_pa"]].copy()

    # Lineup: whoever occupied each slot at its first plate appearance.
    d = d.sort_values(["game_pk", "at_bat_number"])
    lineups = (d.drop_duplicates(["game_pk", "batting_team", "lineup_slot"],
                                 keep="first")
               [["game_pk", "batting_team", "lineup_slot", "batter", "stand",
                 "batting_is_home"]])

    starters = (d[d["is_starter"]]
                .drop_duplicates(["game_pk", "fielding_team"], keep="first")
                [["game_pk", "fielding_team", "pitcher", "p_throws"]])

    # Defensive alignment: the most common occupant of each position over the
    # game, so a late defensive replacement does not displace the man who
    # played eight innings there.
    fcols = [f"fielder_{i}" for i in range(2, 10)]
    have_fielders = all(c in d.columns for c in fcols)
    alignment = {}
    if have_fielders:
        for (gp, team), grp in d.groupby(["game_pk", "fielding_team"]):
            ids = []
            for c in fcols:
                mode = grp[c].dropna()
                if len(mode):
                    ids.append(int(mode.mode().iloc[0]))
            alignment[(gp, team)] = ids

    meta = d.groupby("game_pk").agg(
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        park=("park", "first"), game_date=("game_date", "first")).reset_index()

    finals = (d.groupby(["game_pk", "batting_team"])["runs_pa"].sum()
              .reset_index().rename(columns={"runs_pa": "runs"}))

    games = []
    for gp, m in meta.set_index("game_pk").iterrows():
        lu = lineups[lineups["game_pk"] == gp]
        st = starters[starters["game_pk"] == gp]
        fi = finals[finals["game_pk"] == gp]

        away_lu = lu[lu["batting_team"] == m["away_team"]].sort_values("lineup_slot")
        home_lu = lu[lu["batting_team"] == m["home_team"]].sort_values("lineup_slot")
        away_sp = st[st["fielding_team"] == m["away_team"]]
        home_sp = st[st["fielding_team"] == m["home_team"]]

        if (len(away_lu) != 9 or len(home_lu) != 9
                or len(away_sp) != 1 or len(home_sp) != 1 or len(fi) != 2):
            continue

        ar = fi[fi["batting_team"] == m["away_team"]]["runs"].iloc[0]
        hr = fi[fi["batting_team"] == m["home_team"]]["runs"].iloc[0]

        games.append(dict(
            game_pk=gp, park=m["park"], game_date=m["game_date"],
            home_team=m["home_team"], away_team=m["away_team"],
            away_lineup=away_lu["batter"].to_numpy(),
            home_lineup=home_lu["batter"].to_numpy(),
            away_stands=away_lu["stand"].to_numpy(),
            home_stands=home_lu["stand"].to_numpy(),
            away_starter=int(away_sp["pitcher"].iloc[0]),
            home_starter=int(home_sp["pitcher"].iloc[0]),
            away_hand=away_sp["p_throws"].iloc[0],
            home_hand=home_sp["p_throws"].iloc[0],
            away_fielders=alignment.get((gp, m["away_team"])),
            home_fielders=alignment.get((gp, m["home_team"])),
            away_runs=float(ar), home_runs=float(hr)))
    return games


def run(season, n_sims, max_games, seed, dispersion_sigma=0.0,
        elo_inject=False, elo_strength=1.0, out_suffix="", expand=1.0):
    rng = np.random.default_rng(seed)
    pa = om.load_pa()

    oaa, speed = pr.load_ratings()
    speed_ratings = pr.SpeedRatings(speed, season) if len(speed) else None

    # Team defense, aggregated over whoever is actually on the field. Requires
    # the fielder_* alignment columns; skipped with a warning if the raw cache
    # predates them so the rest of the pipeline still runs.
    defense = None
    try:
        exposure = pr.defensive_exposure(pa)
        defense = pr.DefenseRatings(oaa, exposure, season)
        print(f"defense ratings for {len(defense.rate):,} players")
    except SystemExit as exc:
        print(f"defense DISABLED: {exc}")

    # Transitions are counted from the two most recent prior seasons, not
    # every prior season.
    #
    # The tables encode the run environment they were counted in: a single in a
    # high-scoring year converted to runs at that year's rate, and applying
    # that to a lower-scoring season inflates everything downstream. Building
    # 2024 off 2021-2023 meant a prior averaging 4.474 runs per team-game
    # against 2024's actual 4.392, and the simulator over-produced runs by 4.2%
    # as a result -- invisible in win probability, which cancels, but a direct
    # bias on totals.
    #
    # Two seasons keeps roughly 350,000 plate appearances, which is ample for
    # 1,728 cells with hierarchical backoff, while tracking the environment
    # closely enough to matter.
    trans_seasons = [s for s in sorted(pa["season"].unique())
                     if season - 2 <= s < season]
    print(f"building transition tables from {trans_seasons}...")
    table = baserunning.build(
        pa[pa["season"].isin(trans_seasons)], seasons=None, min_count=25,
        speed_ratings=speed_ratings)

    tables = om.build_count_tables(pa)
    om.TTO_MULT = om.fit_tto(pa[pa["season"] < season])
    tto_mult = np.stack([om.TTO_MULT[t] for t in [1, 2, 3, 4]])
    park_factors = om.fit_park_factors(pa, season)
    calib = om.fit_calibration(tables, pa, season - 1,
                               park_factors=om.fit_park_factors(pa, season - 1))
    reliever_vec = sim.league_reliever_vectors(pa[pa["season"] < season])
    home_adv = om.fit_home_advantage(pa, season)
    print("home advantage multipliers: "
          + "  ".join(f"{o} {home_adv['home'][i]:.4f}"
                      for i, o in enumerate(OUTCOMES)))

    games = extract_games(pa, season)
    print(f"{len(games):,} complete games in {season}")
    if max_games and len(games) > max_games:
        idx = rng.choice(len(games), max_games, replace=False)
        games = [games[i] for i in sorted(idx)]
        print(f"  sampled {len(games):,}")

    # One as-of model per month, so nothing later in the season leaks back.
    months = sorted({g["game_date"].month for g in games})
    models = {}
    for month in months:
        done = pa[(pa["season"] == season)
                  & (pd.to_datetime(pa["game_date"]).dt.month < month)]
        std = om.build_count_tables(done) if len(done) else None
        models[month] = om.OutcomeModel(tables, season, season_to_date=std)
    print(f"built {len(models)} as-of models")

    sim_away, sim_home = [], []
    act_away, act_home = [], []
    p_home = []

    t0 = time.time()
    lw = baserunning.linear_weights(
        pa[pa["season"] < season],
        baserunning.run_expectancy(pa[pa["season"] < season]))

    # --- team-strength expansion (compression fix) ------------------------
    # Summing individually-regressed player projections under-states how
    # different teams are: regressing actual team run differential on predicted
    # gives a slope of 1.46, so a one-run predicted edge is worth 1.46 actual
    # runs. Expanding each team's own strength away from league average
    # corrects that. Deliberately uses NO Elo -- the injection experiment
    # showed that borrowing Elo's ordering destroys the independence that makes
    # the simulator worth blending in the first place.
    expand_delta = {}
    if expand != 1.0:
        meta_x = (pa[pa["is_pa"] & (pa["season"] == season)].groupby("game_pk")
                  .agg(home_team=("home_team", "first"),
                       away_team=("away_team", "first")).reset_index())
        prior_path = os.path.join(config.PROCESSED_DIR, f"sim_preds_{season}.csv")
        if os.path.exists(prior_path):
            s_prev = ts.team_strength_table(pd.read_csv(prior_path), meta_x)
            for row in meta_x.itertuples(index=False):
                expand_delta[row.game_pk] = (
                    (expand - 1.0) * s_prev.get(row.away_team, 0.0),
                    (expand - 1.0) * s_prev.get(row.home_team, 0.0))
            print(f"team-strength expansion x{expand:.2f} applied to "
                  f"{len(expand_delta):,} games "
                  f"(team strength sd {s_prev.std():.4f} runs/game)")
        else:
            print(f"no prior run at {prior_path}; expansion disabled")

    # --- Elo team-strength injection -------------------------------------
    # Blend weights are fitted on the season before the holdout, and the
    # simulator's own strength estimate comes from its previous uncorrected
    # run -- a deterministic function of pre-game inputs, so this refines a
    # fixed point rather than leaking anything.
    elo_delta = {}
    if elo_inject:
        elo = ts.elo_by_game(pa)
        prior = [s for s in sorted(pa["season"].unique()) if s < season]
        runs_per_elo = ts.fit_elo_to_runs(pa, elo, prior)

        meta = (pa[pa["is_pa"]].groupby("game_pk")
                .agg(season=("season", "first"),
                     home_team=("home_team", "first"),
                     away_team=("away_team", "first")).reset_index())

        def strengths(yr):
            path = os.path.join(config.PROCESSED_DIR, f"sim_preds_{yr}.csv")
            if not os.path.exists(path):
                return None, None, None
            sp = pd.read_csv(path)
            mm = meta[meta["season"] == yr]
            s = ts.team_strength_table(sp, mm[["game_pk", "home_team", "away_team"]])
            e = elo[elo["season"] == yr]
            estr = pd.concat([
                e.groupby("home_team")["elo_home"].mean(),
                e.groupby("away_team")["elo_away"].mean()], axis=1).mean(axis=1)
            estr = (estr - ts.ELO_START) * runs_per_elo
            rr = (pa[pa["is_pa"] & (pa["season"] == yr)]
                  .groupby(["game_pk", "batting_team"])["runs_pa"].sum().reset_index())
            act = {}
            for t in s.index:
                scored = rr.loc[rr["batting_team"] == t, "runs_pa"].sum()
                gp = mm[(mm["home_team"] == t) | (mm["away_team"] == t)]["game_pk"]
                allowed = rr[rr["game_pk"].isin(gp) & (rr["batting_team"] != t)]["runs_pa"].sum()
                act[t] = (scored - allowed) / max(len(gp), 1)
            return s, estr, pd.Series(act)

        s_prev, e_prev, a_prev = strengths(season - 1)
        if s_prev is not None:
            w_elo, w_sim = ts.fit_blend(s_prev, e_prev, a_prev)
        else:
            w_elo, w_sim = 0.7, 0.3
        print(f"Elo injection: {runs_per_elo:.5f} runs per Elo point; "
              f"blend {w_elo:.2f} Elo / {w_sim:.2f} simulator "
              f"(fitted on {season - 1})")

        s_cur, _, _ = strengths(season)
        if s_cur is None:
            print("  no prior simulator run for this season; injection disabled")
            elo_inject = False
        else:
            eg = elo[elo["season"] == season].set_index("game_pk")
            for gp_ in eg.index:
                row = eg.loc[gp_]
                sh = s_cur.get(row["home_team"], 0.0)
                sa = s_cur.get(row["away_team"], 0.0)
                elo_delta[gp_] = ts.runs_adjustment(
                    row["elo_home"], row["elo_away"], sh, sa,
                    runs_per_elo, w_elo, strength=elo_strength)

    # Hook model and bullpen rosters. Trained on seasons before the holdout so
    # the removal behaviour is learned, not leaked.
    train_seasons = [s for s in sorted(pa["season"].unique()) if s < season]
    hook_model, hook_fr = bullpen.fit_hook_model(pa, seasons=train_seasons)
    hook = sim.HazardHook(hook_model, bullpen.HOOK_FEATURES)
    print(f"hook model fit on {train_seasons}: "
          f"{len(hook_fr):,} decisions, removal rate "
          f"{hook_fr['removed'].mean() * 100:.2f}%")

    rosters = bullpen.BullpenRosters(bullpen.relief_appearances(pa))
    hands = (pa[pa["is_pa"]].drop_duplicates("pitcher")
             .set_index("pitcher")["p_throws"].to_dict())
    rest_lookup = (pa[pa["is_pa"] & pa["is_starter"]]
                   .groupby(["game_pk", "fielding_team"])
                   ["pitcher_days_since_prev_game"].first())
    workload = bullpen.starter_workload_history(pa).set_index(
        ["game_pk", "pitcher"])[["expected_outs", "is_opener"]]

    for i, g in enumerate(games):
        model = models[g["game_date"].month]

        defense_h = None
        if defense is not None and g.get("away_fielders") is not None:
            # Convert each side's summed fielding rate into the fraction of
            # balls-in-play hits it turns into outs, given the run environment.
            defense_h = [
                pr.defense_hit_suppression(
                    defense.team_rate(g["away_fielders"]), model.league, lw),
                pr.defense_hit_suppression(
                    defense.team_rate(g["home_fielders"]), model.league, lw)]

        pen_arms = []
        pen_hand = np.zeros((2, sim.MAX_PITCHERS), dtype=np.int8)
        for ti, team_name in enumerate((g["away_team"], g["home_team"])):
            ids, w, avail, entry = rosters.as_of(
                team_name, g["game_date"], with_entry=True)
            seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
            seq_hands = [hands.get(int(p), "R") for p in seq]
            pen_arms.append((seq, seq_hands))
            # Slot 0 is the starter; relief arms occupy 1 onward.
            for k, hd in enumerate(seq_hands[:sim.MAX_PITCHERS - 1]):
                pen_hand[ti, k + 1] = 1 if hd == "L" else 0

        cum = sim.build_cum_probs(
            model, g["away_lineup"], g["home_lineup"], g["away_stands"],
            g["home_stands"], g["away_starter"], g["home_starter"],
            g["away_hand"], g["home_hand"], reliever_vec,
            park_mult=park_factors.get(g["park"]), calibration=calib,
            home_advantage=home_adv, defense_h=defense_h,
            bullpen_arms=pen_arms,
            team_runs_delta=(elo_delta.get(g["game_pk"])
                             or expand_delta.get(g["game_pk"])),
            linear_weights=lw if (elo_inject or expand != 1.0) else None)

        lineup_speed = None
        if speed_ratings is not None:
            lineup_speed = np.array([
                [speed_ratings.get(int(p)) for p in g["away_lineup"]],
                [speed_ratings.get(int(p)) for p in g["home_lineup"]]],
                dtype=np.float32)

        starter_rest = [
            float(rest_lookup.get((g["game_pk"], g["away_team"]), 5.0) or 5.0),
            float(rest_lookup.get((g["game_pk"], g["home_team"]), 5.0) or 5.0)]

        lineup_side = np.array([
            [1 if s == "L" else 0 for s in g["away_stands"]],
            [1 if s == "L" else 0 for s in g["home_stands"]]], dtype=np.int8)

        exp_outs, is_open = [], []
        for sp in (g["away_starter"], g["home_starter"]):
            key = (g["game_pk"], sp)
            if key in workload.index:
                row = workload.loc[key]
                exp_outs.append(float(row["expected_outs"]))
                is_open.append(float(row["is_opener"]))
            else:
                exp_outs.append(15.5)
                is_open.append(0.0)

        a, h, _ = sim.simulate(cum, table, n_sims, rng, tto_mult=tto_mult,
                               lineup_speed=lineup_speed, hook=hook,
                               starter_rest=starter_rest,
                               starter_expected_outs=exp_outs,
                               starter_is_opener=is_open,
                               dispersion_sigma=dispersion_sigma,
                               pen_hand=pen_hand, lineup_side=lineup_side)
        sim_away.append(a)
        sim_home.append(h)
        act_away.append(g["away_runs"])
        act_home.append(g["home_runs"])
        p_home.append(float((h > a).mean() + 0.5 * (h == a).mean()))

        if (i + 1) % 200 == 0:
            print(f"  {i + 1:,}/{len(games):,} games "
                  f"({time.time() - t0:.0f}s)")

    sim_away = np.array(sim_away)
    sim_home = np.array(sim_home)
    act_away = np.array(act_away)
    act_home = np.array(act_home)
    p_home = np.array(p_home)

    preds = pd.DataFrame({
        "game_pk": [g["game_pk"] for g in games],
        "season": season,
        "p_home": p_home,
        "home_won": (act_home > act_away).astype(float),
        "sim_home_runs": sim_home.mean(axis=1),
        "sim_away_runs": sim_away.mean(axis=1),
        "act_home_runs": act_home,
        "act_away_runs": act_away,
    })
    out = os.path.join(config.PROCESSED_DIR,
                       f"sim_preds_{season}{out_suffix}.csv")
    preds.to_csv(out, index=False)
    print(f"\nwrote per-game predictions -> {out}")

    report(sim_away, sim_home, act_away, act_home, p_home, n_sims,
           time.time() - t0)


def report(sim_away, sim_home, act_away, act_home, p_home, n_sims, elapsed):
    print(f"\nsimulated {sim_away.shape[0]:,} games x {n_sims:,} in "
          f"{elapsed:.0f}s")

    sim_team = np.concatenate([sim_away.ravel(), sim_home.ravel()])
    act_team = np.concatenate([act_away, act_home])

    print("\n--- run distribution, per team-game ---")
    print(f"  mean      simulated {sim_team.mean():6.3f}   actual {act_team.mean():6.3f}")
    print(f"  std dev   simulated {sim_team.std():6.3f}   actual {act_team.std():6.3f}"
          f"   ratio {sim_team.std() / act_team.std():.4f}")

    sim_tot = (sim_away + sim_home).ravel()
    act_tot = act_away + act_home
    print(f"  total runs per game: simulated {sim_tot.mean():6.3f}   "
          f"actual {act_tot.mean():6.3f}")
    print(f"  total runs std dev:  simulated {sim_tot.std():6.3f}   "
          f"actual {act_tot.std():6.3f}   "
          f"ratio {sim_tot.std() / act_tot.std():.4f}")

    print("\n  team runs histogram (% of team-games):")
    print(f"    {'runs':>5} {'simulated':>10} {'actual':>10}")
    for r in range(0, 13):
        s = (sim_team == r).mean() * 100
        a = (act_team == r).mean() * 100
        label = f"{r}" if r < 12 else "12+"
        if r == 12:
            s = (sim_team >= r).mean() * 100
            a = (act_team >= r).mean() * 100
        print(f"    {label:>5} {s:>9.2f}% {a:>9.2f}%")

    print("\n--- game winner ---")
    home_won = (act_home > act_away).astype(float)
    print(f"  actual home win rate    {home_won.mean() * 100:.2f}%")
    print(f"  simulated home win rate {p_home.mean() * 100:.2f}%")

    eps = 1e-9
    p = np.clip(p_home, eps, 1 - eps)
    ll = -np.mean(home_won * np.log(p) + (1 - home_won) * np.log(1 - p))
    brier = np.mean((p - home_won) ** 2)

    base_rate = home_won.mean()
    ll_base = -np.mean(home_won * np.log(base_rate)
                       + (1 - home_won) * np.log(1 - base_rate))
    brier_base = np.mean((base_rate - home_won) ** 2)

    print(f"  model     log loss {ll:.5f}   Brier {brier:.5f}")
    print(f"  home-rate log loss {ll_base:.5f}   Brier {brier_base:.5f}")
    print(f"  accuracy  {(((p > 0.5) == (home_won > 0.5)).mean() * 100):.2f}%   "
          f"(always-home {max(base_rate, 1 - base_rate) * 100:.2f}%)")

    print("\n  win probability calibration:")
    bins = [0, .35, .40, .45, .50, .55, .60, .65, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_home >= lo) & (p_home < hi)
        if m.sum() < 10:
            continue
        print(f"    {lo:.2f}-{hi:.2f}: predicted {p_home[m].mean() * 100:5.1f}%  "
              f"actual {home_won[m].mean() * 100:5.1f}%  (n={m.sum():,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    # Shipped defaults. Expansion 1.2 was selected independently on 2024 and
    # 2025 and won cleanly on both, so it reflects something real. Dispersion
    # is a compromise: 2024 wants 0.10, 2025 wants 0.19, and no single value
    # calibrates both -- how much shared shock the simulation needs depends on
    # how variable that season's scoring actually was. 0.15 splits the
    # difference and is disclosed as such rather than presented as optimal.
    ap.add_argument("--dispersion", type=float, default=0.15,
                    help="per-team lognormal shock sigma (season-unstable: "
                         "2024 prefers 0.10, 2025 prefers 0.19)")
    ap.add_argument("--elo-inject", action="store_true",
                    help="blend Elo team strength into the talent layer")
    ap.add_argument("--elo-strength", type=float, default=1.0,
                    help="scale the Elo correction (1.0 = full blend)")
    ap.add_argument("--suffix", default="",
                    help="suffix for the predictions file, to avoid clobbering")
    ap.add_argument("--expand", type=float, default=1.2,
                    help="expand team-strength differentials (1.0 = off); "
                         "1.2 selected independently on 2024 and 2025")
    ap.add_argument("--params", default="",
                    help="JSON file of tuned outcome-model constants")
    args = ap.parse_args()

    if args.params:
        import json
        with open(args.params) as fh:
            blob = json.load(fh)
        om.ACTIVE_PARAMS = blob.get("final_params", blob)
        print(f"using tuned constants from {args.params}")

    run(args.season, args.sims, args.games, args.seed, args.dispersion,
        args.elo_inject, args.elo_strength, args.suffix, args.expand)


if __name__ == "__main__":
    main()
