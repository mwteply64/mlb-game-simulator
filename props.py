"""Player prop distributions, and whether they are any good.

A prop line only asks which side of a number the result lands on, which is a
much easier question than predicting the number. That is where a simulator has
a structural advantage over a rate estimate: a starter's strikeout total
depends jointly on how many batters he faces and how often he misses bats, and
those two are *coupled* -- a start that ends in the fourth cannot also produce
nine strikeouts. Multiplying a strikeout rate by an expected batters-faced
figure gets the mean about right and the spread badly wrong, which is precisely
what a line-versus-distribution question is sensitive to.

The simulator draws both from the same run, so the dependence comes for free.

    python props.py --season 2025 --starts 400
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
import simulate as sim


def prop_distribution(diag, key, team):
    """Counts across simulations for one side's starter."""
    return diag[key][:, team]


def over_probability(samples, line):
    """Probability of clearing a line. Half-run lines cannot push."""
    return float((samples > line).mean())


def summarize(samples, lines=()):
    out = {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "sd": float(samples.std()),
        "p10": float(np.quantile(samples, 0.1)),
        "p90": float(np.quantile(samples, 0.9)),
    }
    for ln in lines:
        out[f"over_{ln}"] = over_probability(samples, ln)
    return out


class PropSimulator:
    """Everything needed to simulate a slate, built once."""

    def __init__(self, pa, season):
        self.pa = pa
        self.season = season
        prior = pa[pa["season"] < season]
        oaa, speed = pr.load_ratings()
        self.speed_ratings = pr.SpeedRatings(speed, season)
        self.table = baserunning.build(prior, min_count=25,
                                       speed_ratings=self.speed_ratings)
        tables = om.build_count_tables(pa)
        self.tables = tables
        om.TTO_MULT = om.fit_tto(prior)
        self.tto_mult = np.stack([om.TTO_MULT[t] for t in (1, 2, 3, 4)])
        self.park_factors = om.fit_park_factors(pa, season)
        self.calib = om.fit_calibration(
            tables, pa, season - 1,
            park_factors=om.fit_park_factors(pa, season - 1))
        self.home_adv = om.fit_home_advantage(pa, season)
        self.reliever_vec = sim.league_reliever_vectors(prior)

        # Pinch hitting. Without it the simulator runs a fixed nine-slot
        # lineup, which overstates a starter's plate appearances by about 2.6%
        # and inflates every hitter counting prop by the same margin.
        import leverage as _lev
        import pinch_hitting as _ph
        self.pinch_vec = _ph.pinch_hitter_vectors(prior)
        _h = _ph.load_hazard()
        self.pinch = ((_h[1], _lev.load_leverage_table()) if _h else None)
        self.hook_model, _ = bullpen.fit_hook_model(
            pa, seasons=sorted(prior["season"].unique()))
        self.hook = sim.HazardHook(self.hook_model, bullpen.HOOK_FEATURES)
        self.rosters = bullpen.BullpenRosters(bullpen.relief_appearances(pa))
        self.hands = (pa[pa["is_pa"]].drop_duplicates("pitcher")
                      .set_index("pitcher")["p_throws"].to_dict())
        self.workload = bullpen.starter_workload_history(pa).set_index(
            ["game_pk", "pitcher"])[["expected_outs", "is_opener"]]

        self.models = {}
        for month in range(3, 11):
            done = pa[(pa["season"] == season)
                      & (pd.to_datetime(pa["game_date"]).dt.month < month)]
            std = om.build_count_tables(done) if len(done) else None
            self.models[month] = om.OutcomeModel(tables, season,
                                                 season_to_date=std)

    def run(self, game, n_sims, rng, hook_sigma=0.0):
        model = self.models[game["game_date"].month]
        pen_arms, pen_hand = [], np.zeros((2, sim.MAX_PITCHERS), dtype=np.int8)
        for ti, team in enumerate((game["away_team"], game["home_team"])):
            ids, w, avail, entry = self.rosters.as_of(
                team, game["game_date"], with_entry=True)
            seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
            sh = [self.hands.get(int(p), "R") for p in seq]
            pen_arms.append((seq, sh))
            for k, hd in enumerate(sh[:sim.MAX_PITCHERS - 1]):
                pen_hand[ti, k + 1] = 1 if hd == "L" else 0

        cum = sim.build_cum_probs(
            model, game["away_lineup"], game["home_lineup"],
            game["away_stands"], game["home_stands"], game["away_starter"],
            game["home_starter"], game["away_hand"], game["home_hand"],
            self.reliever_vec, park_mult=self.park_factors.get(game["park"]),
            calibration=self.calib, home_advantage=self.home_adv,
            bullpen_arms=pen_arms, pinch_vec=self.pinch_vec)

        lineup_speed = np.array(
            [[self.speed_ratings.get(int(p)) for p in game["away_lineup"]],
             [self.speed_ratings.get(int(p)) for p in game["home_lineup"]]],
            dtype=np.float32)
        lineup_side = np.array(
            [[1 if s == "L" else 0 for s in game["away_stands"]],
             [1 if s == "L" else 0 for s in game["home_stands"]]],
            dtype=np.int8)

        eo, op = [], []
        for spid in (game["away_starter"], game["home_starter"]):
            key = (game["game_pk"], spid)
            if key in self.workload.index:
                eo.append(float(self.workload.loc[key, "expected_outs"]))
                op.append(float(self.workload.loc[key, "is_opener"]))
            else:
                eo.append(15.5)
                op.append(0.0)

        _, _, _, diag = sim.simulate(
            cum, self.table, n_sims, rng, tto_mult=self.tto_mult,
            lineup_speed=lineup_speed, hook=self.hook,
            starter_expected_outs=eo, starter_is_opener=op,
            dispersion_sigma=0.19, pen_hand=pen_hand, pinch=self.pinch,
            lineup_side=lineup_side, diagnostics=True,
            hook_sigma=hook_sigma)
        return diag


def actual_batter_stats(pa, season):
    """Per batter per game: the counting lines hitter props are priced on.

    Singles through home runs and walks come straight from the outcome column.
    Runs and runs batted in are reconstructed with the same lead-runner-first
    convention the simulator uses, so the two are measured consistently -- they
    will differ slightly from official scoring, which does not credit a run
    batted in on an error or a double play.
    """
    d = pa[pa["is_pa"] & (pa["season"] == season)].copy()
    d = d.sort_values(["game_pk", "at_bat_number"])

    counts = (d.pivot_table(index=["game_pk", "batter"], columns="outcome",
                            values="is_pa", aggfunc="size", fill_value=0)
              .reset_index())
    for c in ("1B", "2B", "3B", "HR", "BB"):
        if c not in counts.columns:
            counts[c] = 0

    counts["hits"] = counts[["1B", "2B", "3B", "HR"]].sum(axis=1)
    counts["total_bases"] = (counts["1B"] + 2 * counts["2B"]
                             + 3 * counts["3B"] + 4 * counts["HR"])

    rbi = (d.groupby(["game_pk", "batter"])["runs_pa"].sum()
           .rename("rbi").reset_index())
    counts = counts.merge(rbi, on=["game_pk", "batter"], how="left")

    # Runs scored: walk each half-inning and attribute crossings lead-first.
    runs = {}
    for (_, _), grp in d.groupby(["game_pk", "half_key"], sort=False):
        on = [-1, -1, -1]
        for r in grp.itertuples(index=False):
            before = int(r.base_code_before)
            n_runs = int(min(max(r.runs_pa, 0), 4))
            order = [i for i in (2, 1, 0) if (before >> i) & 1]
            for base in order[:min(n_runs, len(order))]:
                if on[base] >= 0:
                    runs[(r.game_pk, on[base])] = runs.get(
                        (r.game_pk, on[base]), 0) + 1
                on[base] = -1
            if n_runs > len(order):
                runs[(r.game_pk, r.batter)] = runs.get(
                    (r.game_pk, r.batter), 0) + 1
            after = int(r.base_code_after) if r.base_code_after >= 0 else 0
            survivors = [on[i] for i in order[min(n_runs, len(order)):]]
            new_on = [-1, -1, -1]
            after_bases = [i for i in (2, 1, 0) if (after >> i) & 1]
            reached = r.outcome in ("BB", "HBP", "1B", "2B", "3B")
            slots = after_bases[:-1] if (reached and after_bases) else after_bases
            if len(survivors) > len(slots):
                survivors = survivors[len(survivors) - len(slots):]
            for k, base in enumerate(slots):
                if k < len(survivors):
                    new_on[base] = survivors[k]
            if reached and after_bases:
                new_on[after_bases[-1]] = r.batter
            on = new_on

    rs = pd.Series(runs).rename("runs")
    if len(rs):
        rs.index = pd.MultiIndex.from_tuples(rs.index,
                                             names=["game_pk", "batter"])
        counts = counts.merge(rs.reset_index(), on=["game_pk", "batter"],
                              how="left")
    counts["runs"] = counts.get("runs", pd.Series(0.0)).fillna(0.0)
    counts["hrr"] = counts["hits"] + counts["runs"] + counts["rbi"]
    counts["fantasy"] = (3 * counts["1B"] + 5 * counts["2B"]
                         + 8 * counts["3B"] + 10 * counts["HR"]
                         + 2 * counts["runs"] + 2 * counts["rbi"]
                         + 2 * counts["BB"])
    return counts


def calibration_report(df, pred_col, act_col, lines, label):
    print(f"\n{label}")
    print(f"  predicted mean {df[pred_col].mean():7.3f}   "
          f"actual mean {df[act_col].mean():7.3f}   "
          f"correlation {df[pred_col].corr(df[act_col]):+.4f}")
    err = df[pred_col] - df[act_col]
    naive = df[act_col].mean() - df[act_col]
    print(f"  RMSE {np.sqrt((err ** 2).mean()):.3f}  vs "
          f"{np.sqrt((naive ** 2).mean()):.3f} for the mean")
    for ln in lines:
        col = f"p_over_{label}_{ln}"
        if col not in df.columns:
            continue
        hit = (df[act_col] > ln).astype(float)
        print(f"    line {ln:<5} predicted {df[col].mean() * 100:5.1f}%   "
              f"actual {hit.mean() * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--starts", type=int, default=400)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pa = om.load_pa()
    ps = PropSimulator(pa, args.season)
    games = evaluate_sim.extract_games(pa, args.season)
    rng.shuffle(games)
    games = games[:args.starts]
    print(f"simulating {len(games):,} games x {args.sims:,}\n")

    # Actual starter lines for the same games.
    d = pa[pa["is_pa"] & (pa["season"] == args.season)]
    actual = (d[d["is_starter"]]
              .groupby(["game_pk", "fielding_team", "pitcher"])
              .agg(k=("outcome", lambda s: (s == "K").sum()),
                   outs=("outs_pa", "sum"),
                   bf=("is_pa", "size")).reset_index())
    act_idx = actual.set_index(["game_pk", "pitcher"])

    bat_actual = actual_batter_stats(pa, args.season).set_index(
        ["game_pk", "batter"])

    rows, brows = [], []
    for i, g in enumerate(games):
        diag = ps.run(g, args.sims, rng)
        for ti, (team, spid, lineup) in enumerate(
                ((g["away_team"], g["away_starter"], g["away_lineup"]),
                 (g["home_team"], g["home_starter"], g["home_lineup"]))):
            key = (g["game_pk"], spid)
            if key in act_idx.index:
                a = act_idx.loc[key]
                k_s = diag["starter_k"][:, ti]
                o_s = diag["starter_outs"][:, ti]
                r_s = diag["starter_runs"][:, ti]
                w_s = diag["starter_bb"][:, ti]
                p_s = diag["starter_pitches"][:, ti]
                rows.append({
                    "game_pk": g["game_pk"], "pitcher": spid,
                    "pred_k": float(k_s.mean()), "act_k": int(a["k"]),
                    "pred_outs": float(o_s.mean()), "act_outs": float(a["outs"]),
                    "pred_runs": float(r_s.mean()),
                    "pred_bb": float(w_s.mean()),
                    "pred_pitches": float(p_s.mean()),
                    **{f"p_over_K_{ln}": over_probability(k_s, ln)
                       for ln in (3.5, 4.5, 5.5, 6.5)},
                    **{f"p_over_outs_{ln}": over_probability(o_s, ln)
                       for ln in (14.5, 17.5)},
                })

            # Hitters, by lineup slot.
            for slot in range(9):
                bid = int(lineup[slot])
                bkey = (g["game_pk"], bid)
                if bkey not in bat_actual.index:
                    continue
                ba = bat_actual.loc[bkey]
                tb = diag["bat_total_bases"][:, ti, slot]
                hrr = diag["bat_hrr"][:, ti, slot]
                fan = diag["bat_fantasy"][:, ti, slot]
                brows.append({
                    "game_pk": g["game_pk"], "batter": bid, "slot": slot,
                    "pred_tb": float(tb.mean()), "act_tb": float(ba["total_bases"]),
                    "pred_hrr": float(hrr.mean()), "act_hrr": float(ba["hrr"]),
                    "pred_fan": float(fan.mean()), "act_fan": float(ba["fantasy"]),
                    **{f"p_over_TB_{ln}": over_probability(tb, ln)
                       for ln in (0.5, 1.5, 2.5)},
                    **{f"p_over_HRR_{ln}": over_probability(hrr, ln)
                       for ln in (0.5, 1.5, 2.5)},
                })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(games)}")

    r = pd.DataFrame(rows)
    b = pd.DataFrame(brows)
    print(f"\n{len(r):,} starts, {len(b):,} batter-games\n")

    print("=" * 62)
    print("PITCHING PROPS")
    print("=" * 62)
    calibration_report(r, "pred_k", "act_k", (3.5, 4.5, 5.5, 6.5), "K")
    calibration_report(r, "pred_outs", "act_outs", (14.5, 17.5), "outs")
    print(f"\nruns allowed: predicted mean {r['pred_runs'].mean():.3f}")
    print(f"walks:        predicted mean {r['pred_bb'].mean():.3f}")
    print(f"pitches:      predicted mean {r['pred_pitches'].mean():.1f}")

    print("\n  strikeout calibration by predicted probability, line 4.5:")
    hit = (r["act_k"] > 4.5).astype(float)
    for lo, hi in ((0, .25), (.25, .40), (.40, .60), (.60, .75), (.75, 1.01)):
        m = (r["p_over_K_4.5"] >= lo) & (r["p_over_K_4.5"] < hi)
        if m.sum() >= 25:
            print(f"    {lo:.2f}-{hi:.2f}: model "
                  f"{r.loc[m, 'p_over_K_4.5'].mean() * 100:5.1f}%  actual "
                  f"{hit[m].mean() * 100:5.1f}%  (n={m.sum():,})")

    print("\n" + "=" * 62)
    print("HITTER PROPS")
    print("=" * 62)
    calibration_report(b, "pred_tb", "act_tb", (0.5, 1.5, 2.5), "TB")
    calibration_report(b, "pred_hrr", "act_hrr", (0.5, 1.5, 2.5), "HRR")
    print(f"\nfantasy points  predicted mean {b['pred_fan'].mean():6.3f}   "
          f"actual {b['act_fan'].mean():6.3f}   "
          f"correlation {b['pred_fan'].corr(b['act_fan']):+.4f}")
    ferr = b["pred_fan"] - b["act_fan"]
    fnaive = b["act_fan"].mean() - b["act_fan"]
    print(f"                RMSE {np.sqrt((ferr ** 2).mean()):.3f}  vs "
          f"{np.sqrt((fnaive ** 2).mean()):.3f} for the mean")

    print("\n  total bases calibration, line 1.5:")
    hit = (b["act_tb"] > 1.5).astype(float)
    for lo, hi in ((0, .15), (.15, .25), (.25, .35), (.35, 1.01)):
        m = (b["p_over_TB_1.5"] >= lo) & (b["p_over_TB_1.5"] < hi)
        if m.sum() >= 40:
            print(f"    {lo:.2f}-{hi:.2f}: model "
                  f"{b.loc[m, 'p_over_TB_1.5'].mean() * 100:5.1f}%  actual "
                  f"{hit[m].mean() * 100:5.1f}%  (n={m.sum():,})")

    r.to_csv(f"{config.PROCESSED_DIR}/prop_pitching_{args.season}.csv",
             index=False)
    b.to_csv(f"{config.PROCESSED_DIR}/prop_hitting_{args.season}.csv",
             index=False)
    print(f"\nwrote prop_pitching_{args.season}.csv and "
          f"prop_hitting_{args.season}.csv")


if __name__ == "__main__":
    main()
