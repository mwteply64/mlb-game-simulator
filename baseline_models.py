"""Team-strength baselines the simulator has to beat to justify itself.

Three, in increasing order of difficulty:

1. **Elo** -- record only. Updated on game results with a margin-of-victory
   multiplier, home advantage, and between-season regression to the mean. This
   is deliberately the kind of model the project set out *not* to build: it
   evaluates teams on their record. It is the "is any of this necessary"
   reference.

2. **Run differential** -- season-to-date runs scored and allowed per game.
   Better than record, still purely results-based.

3. **Talent logistic** -- a plain logistic regression fed the *same* projected
   inputs the simulator uses: lineup offensive talent against the opposing
   starter's hand, starting pitcher talent, bullpen talent, all as-of. No
   innings, no base-out states, no sequencing, no times-through-order, no
   bullpen timing.

Baseline 3 is the one that matters. It shares the simulator's entire projection
stack, so the difference between them isolates exactly one thing: whether
simulating game structure is worth its complexity. If the simulator cannot beat
it, the projection layer is doing all the work and the Monte Carlo is
decoration.

    python baseline_models.py --holdout 2025
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import bullpen
import config
import evaluate_sim
import outcome_model as om
import player_ratings as pr

OUTCOMES = config.OUTCOMES


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def score(name, p, y):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    acc = ((p > 0.5) == (y > 0.5)).mean()
    print(f"  {name:<28} log loss {ll:.5f}   Brier {brier:.5f}   "
          f"accuracy {acc * 100:5.2f}%")
    return ll, brier, acc


# --------------------------------------------------------------------------
# 1. Elo
# --------------------------------------------------------------------------

def elo_predictions(pa, holdout, k=4.0, home_adv=24.0, regress=0.25,
                    mov=True):
    """Walk the full history forward, emitting each game's pre-game home win
    probability. Ratings only ever use games already played."""
    games = (pa[pa["is_pa"]]
             .groupby("game_pk")
             .agg(season=("season", "first"), date=("game_date", "first"),
                  home=("home_team", "first"), away=("away_team", "first"))
             .reset_index())
    runs = (pa[pa["is_pa"]].groupby(["game_pk", "batting_team"])["runs_pa"]
            .sum().reset_index())
    runs = runs.pivot(index="game_pk", columns="batting_team",
                      values="runs_pa")

    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    rating = {}
    prev_season = None
    out = []
    for row in games.itertuples(index=False):
        if prev_season is not None and row.season != prev_season:
            # Carry ratings across the offseason but pull them toward the mean.
            for t in rating:
                rating[t] = 1500 + (rating[t] - 1500) * (1 - regress)
        prev_season = row.season

        rh = rating.setdefault(row.home, 1500.0)
        ra = rating.setdefault(row.away, 1500.0)

        diff = rh + home_adv - ra
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        hr = runs.at[row.game_pk, row.home]
        ar = runs.at[row.game_pk, row.away]
        if pd.isna(hr) or pd.isna(ar):
            continue
        result = 1.0 if hr > ar else 0.0

        mult = 1.0
        if mov:
            margin = abs(hr - ar)
            mult = np.log(margin + 1.0)

        delta = k * mult * (result - p_home)
        rating[row.home] = rh + delta
        rating[row.away] = ra - delta

        out.append((row.game_pk, row.season, p_home, result))

    # Every season is returned; ratings at any point used only prior games, so
    # earlier seasons are usable as training features without leakage.
    return pd.DataFrame(out, columns=["game_pk", "season", "p_home", "home_won"])


# --------------------------------------------------------------------------
# 2 & 3. Feature-based models
# --------------------------------------------------------------------------

def run_differential_features(pa, seasons):
    """Season-to-date runs scored and allowed per game, entering each game."""
    g = (pa[pa["is_pa"]]
         .groupby(["game_pk", "season", "game_date", "batting_team",
                   "fielding_team"])["runs_pa"].sum().reset_index())

    g = g.sort_values(["season", "game_date", "game_pk"])
    rows = []
    scored, allowed, played = {}, {}, {}
    cur_season = None
    for r in g.itertuples(index=False):
        if r.season != cur_season:
            scored, allowed, played = {}, {}, {}
            cur_season = r.season
        rows.append((r.game_pk, r.batting_team,
                     scored.get(r.batting_team, 0.0),
                     allowed.get(r.batting_team, 0.0),
                     played.get(r.batting_team, 0)))
        scored[r.batting_team] = scored.get(r.batting_team, 0.0) + r.runs_pa
        allowed[r.fielding_team] = allowed.get(r.fielding_team, 0.0) + r.runs_pa
        played[r.batting_team] = played.get(r.batting_team, 0) + 1

    return pd.DataFrame(rows, columns=["game_pk", "team", "rs", "ra", "gp"])


def talent_features(pa, tables, games, holdout, lw, defense=None,
                    speed_ratings=None):
    """Per-game talent features from the simulator's own projection stack.

    Every quantity is an expected run value per plate appearance, computed with
    empirical linear weights, using an as-of model so nothing later in the
    season leaks backward.
    """
    months = sorted({g["game_date"].month for g in games})
    models = {}
    for month in months:
        done = pa[(pa["season"] == holdout)
                  & (pd.to_datetime(pa["game_date"]).dt.month < month)]
        std = om.build_count_tables(done) if len(done) else None
        models[month] = om.OutcomeModel(tables, holdout, season_to_date=std)

    # Team bullpen talent, as of the game date: the arms that have actually
    # appeared in the last three weeks, weighted by their share of relief
    # batters faced, and projected individually. This replaces a pooled
    # prior-season average whose regression coefficient collapsed to zero
    # whenever Elo was present -- direct evidence it carried nothing current.
    rosters = bullpen.BullpenRosters(bullpen.relief_appearances(pa))
    rel = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES) & ~pa["is_starter"]
             & (pa["season"] < holdout)]
    lg_rel = rel["outcome"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    bp_league = float(np.dot(lg_rel, lw))

    talent_cache = {}

    def reliever_value(month, pid):
        key = (month, int(pid))
        if key not in talent_cache:
            m = models[month]
            talent_cache[key] = float(np.mean([
                np.dot(m.pitcher_vector(int(pid), hand, side), lw)
                for hand in ("L", "R") for side in ("L", "R")]))
        return talent_cache[key]

    def pen_talent(team, date, month):
        ids, w, avail = rosters.as_of(team, date)
        if len(ids) == 0:
            return bp_league
        if avail.any():
            ids, w = ids[avail], w[avail] / w[avail].sum()
        return float(np.dot(w, [reliever_value(month, p) for p in ids]))

    rows = []
    for g in games:
        m = models[g["game_date"].month]

        def lineup_talent(lineup, stands, opp_hand):
            vals = [np.dot(m.batter_vector(int(b), s, opp_hand), lw)
                    for b, s in zip(lineup, stands)]
            return float(np.mean(vals))

        def starter_talent(pid, hand):
            # Against a league-representative mix of batter sides.
            return float(np.mean([
                np.dot(m.pitcher_vector(int(pid), hand, side), lw)
                for side in ["L", "R"]]))

        def team_defense(fielders):
            if defense is None or not fielders:
                return 0.0
            return defense.team_rate(fielders)

        def team_speed(lineup):
            if speed_ratings is None:
                return 0.0
            return float(np.mean([speed_ratings.get(int(p)) for p in lineup]))

        rows.append(dict(
            game_pk=g["game_pk"],
            home_won=1.0 if g["home_runs"] > g["away_runs"] else 0.0,
            off_home=lineup_talent(g["home_lineup"], g["home_stands"], g["away_hand"]),
            off_away=lineup_talent(g["away_lineup"], g["away_stands"], g["home_hand"]),
            sp_home=starter_talent(g["home_starter"], g["home_hand"]),
            sp_away=starter_talent(g["away_starter"], g["away_hand"]),
            bp_home=pen_talent(g["home_team"], g["game_date"],
                               g["game_date"].month),
            bp_away=pen_talent(g["away_team"], g["game_date"],
                               g["game_date"].month),
            def_home=team_defense(g.get("home_fielders")),
            def_away=team_defense(g.get("away_fielders")),
            spd_home=team_speed(g["home_lineup"]),
            spd_away=team_speed(g["away_lineup"]),
        ))
    return pd.DataFrame(rows)


def fit_logistic(X, y, iters=400, lr=0.5, l2=1e-3):
    """Plain gradient-descent logistic regression. Standardized inputs, so a
    handful of features converges quickly without a solver dependency."""
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


def apply_logistic(w, mu, sd, X):
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return 1.0 / (1.0 + np.exp(-Z @ w))


# --------------------------------------------------------------------------

def run(holdout):
    pa = om.load_pa()

    print("deriving run expectancy and linear weights from prior seasons...")
    re = baserunning.run_expectancy(pa[pa["season"] < holdout])
    lw = baserunning.linear_weights(pa[pa["season"] < holdout], re)
    print("  run expectancy, bases empty: "
          + "  ".join(f"{o} out {re[0, o]:.3f}" for o in range(3)))
    print("  run expectancy, loaded:      "
          + "  ".join(f"{o} out {re[7, o]:.3f}" for o in range(3)))
    print("  linear weights: "
          + "  ".join(f"{o} {lw[i]:+.3f}" for i, o in enumerate(OUTCOMES)))

    print("\nextracting games...")
    games_test = evaluate_sim.extract_games(pa, holdout)
    print(f"  {len(games_test):,} in {holdout}")

    y = np.array([1.0 if g["home_runs"] > g["away_runs"] else 0.0
                  for g in games_test])
    print(f"\nholdout {holdout}: {len(y):,} games, home win rate "
          f"{y.mean() * 100:.2f}%")

    print("\n--- baselines ---")
    base_rate = y.mean()
    score("always home", np.full(len(y), base_rate), y)

    # 1. Elo
    elo_all = elo_predictions(pa, holdout).set_index("game_pk")
    order = [g["game_pk"] for g in games_test]
    p_elo = elo_all.reindex(order)["p_home"].fillna(0.5).to_numpy()
    score("Elo (record only)", p_elo, y)

    # 2. Run differential
    rd = run_differential_features(pa, None).set_index(["game_pk", "team"])
    feats = []
    for g in games_test:
        try:
            h = rd.loc[(g["game_pk"], g["home_team"])]
            a = rd.loc[(g["game_pk"], g["away_team"])]
        except KeyError:
            feats.append([0.0, 0.0])
            continue
        hd = (h["rs"] - h["ra"]) / max(h["gp"], 1)
        ad = (a["rs"] - a["ra"]) / max(a["gp"], 1)
        feats.append([hd, ad])
    Xrd = np.array(feats)

    # 3. Talent logistic
    tables = om.build_count_tables(pa)

    oaa, speed = pr.load_ratings()
    speed_ratings = pr.SpeedRatings(speed, holdout) if len(speed) else None
    defense = None
    try:
        defense = pr.DefenseRatings(oaa, pr.defensive_exposure(pa), holdout)
        print(f"\ndefense ratings for {len(defense.rate):,} players")
    except SystemExit as exc:
        print(f"\ndefense DISABLED: {exc}")

    print("building talent features (this is the model that matters)...")
    tf_test = talent_features(pa, tables, games_test, holdout, lw,
                              defense=defense, speed_ratings=speed_ratings)

    train_seasons = [s for s in sorted(pa["season"].unique())
                     if s < holdout and s >= holdout - 3]
    Xtr_list, ytr_list, Xrd_tr, = [], [], []
    for s in train_seasons:
        gs = evaluate_sim.extract_games(pa, s)
        d_s = None
        if defense is not None:
            d_s = pr.DefenseRatings(oaa, pr.defensive_exposure(pa), s)
        tf = talent_features(pa, tables, gs, s, lw, defense=d_s,
                             speed_ratings=(pr.SpeedRatings(speed, s)
                                            if len(speed) else None))
        Xtr_list.append(tf)
        ytr_list.append(tf["home_won"].to_numpy())
        for g in gs:
            try:
                h = rd.loc[(g["game_pk"], g["home_team"])]
                a = rd.loc[(g["game_pk"], g["away_team"])]
                Xrd_tr.append([(h["rs"] - h["ra"]) / max(h["gp"], 1),
                               (a["rs"] - a["ra"]) / max(a["gp"], 1)])
            except KeyError:
                Xrd_tr.append([0.0, 0.0])
        print(f"  trained on {s}: {len(gs):,} games")

    cols = ["off_home", "off_away", "sp_home", "sp_away", "bp_home", "bp_away",
            "def_home", "def_away", "spd_home", "spd_away"]
    Xtr = pd.concat(Xtr_list)[cols].to_numpy()
    ytr = np.concatenate(ytr_list)
    Xte = tf_test[cols].to_numpy()

    Xrd_tr = np.array(Xrd_tr)
    w, mu, sd = fit_logistic(Xrd_tr, ytr)
    score("run differential logistic", apply_logistic(w, mu, sd, Xrd), y)

    w, mu, sd = fit_logistic(Xtr, ytr)
    p_tal = apply_logistic(w, mu, sd, Xte)
    score("talent logistic", p_tal, y)

    # 4. Elo + talent. The question this answers is whether the component
    # projections carry information Elo does not already have. If they add
    # nothing here, the whole component approach is redundant for predicting
    # game winners.
    train_pks = pd.concat(Xtr_list)["game_pk"].to_numpy()
    elo_tr = elo_all.reindex(train_pks)["p_home"].fillna(0.5).to_numpy()
    Xtr_c = np.column_stack([Xtr, np.log(elo_tr / (1 - elo_tr))])
    Xte_c = np.column_stack([Xte, np.log(p_elo / (1 - p_elo))])

    w_c, mu_c, sd_c = fit_logistic(Xtr_c, ytr)
    p_comb = apply_logistic(w_c, mu_c, sd_c, Xte_c)
    score("Elo + talent logistic", p_comb, y)

    # Elo alone, refit on the same training window, so the comparison against
    # the combined model is like-for-like rather than Elo's raw formula.
    w_e, mu_e, sd_e = fit_logistic(elo_tr[:, None], ytr)
    score("Elo refit alone", apply_logistic(w_e, mu_e, sd_e, p_elo[:, None]), y)

    print(f"\n  talent logistic coefficients (standardized):")
    for name, coef in zip(["intercept"] + cols, w):
        print(f"    {name:<12} {coef:+.4f}")
    print(f"\n  Elo + talent coefficients (standardized):")
    for name, coef in zip(["intercept"] + cols + ["elo_logit"], w_c):
        print(f"    {name:<12} {coef:+.4f}")

    print("\n--- simulator, for comparison (from evaluate_sim.py) ---")
    print("  simulator                    log loss 0.68337   Brier 0.24524   "
          "accuracy 54.73%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=2025)
    args = ap.parse_args()
    run(args.holdout)


if __name__ == "__main__":
    main()
