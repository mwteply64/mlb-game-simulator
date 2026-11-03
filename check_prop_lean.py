"""Model versus the posted line, for every prop market we can price.

Covers total bases (hitters), strikeouts and outs recorded (starters). One
simulation per game feeds all three, since they are all read off the same run.

**What this can and cannot measure.** ESPN's propBets feed returns a single
price per prop with no side label -- the item carries `competition, athlete,
provider, type, odds, lastUpdated, current, open`, and `current` holds only the
target number. Which side that price belongs to is not stated, and it is not
consistent across markets: on 2026-08-04 the recorded Total Bases price implied
63.2% against an actual over rate of 35.2% (so it was the under), while Total
Strikeouts implied 52.8% against an actual 50.8% (the over).

So the log-loss complementarity test -- does blending our number with the
market's beat the market alone, the test that settled the moneyline question --
**cannot be computed here.** There is no defined event to attach the market's
probability to. What is computable needs only the line:

* which side the model takes, and by how much
* whether that side wins, against the 52.4% break-even at -110

That is weaker. It says whether the model beats the *number*, not whether it
beats the *price*, and the prices above are not -110. Treat a win rate over
52.4% as a reason to keep recording, never as a measured edge.

Two traps that have already produced fake results in this project, both guarded
here: match `prop_type` exactly (substring matching swept in the milestone
markets and manufactured a 72% win rate that vanished on the real market), and
report the match rate against lines whose games were actually simulated rather
than against every line recorded.

    python check_prop_lean.py --season 2026 --sims 2000
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate_sim
import live_props as lp
import outcome_model as om
import props

# prop_type -> (side, diagnostic key, label). "side" says whose line it is, and
# therefore how to join it: a pitcher line keys on the starter, a hitter line on
# whichever lineup slot the player occupies.
MARKETS = {
    "Total Bases": ("hitter", "bat_total_bases", "total bases"),
    "Total Hits": ("hitter", "bat_hits", "hits"),
    "Total Hits + Runs + RBIs": ("hitter", "bat_hrr", "hits+runs+RBI"),
    "Total Strikeouts": ("pitcher", "starter_k", "strikeouts"),
    "Total Outs Recorded": ("pitcher", "starter_outs", "outs recorded"),
}

BREAK_EVEN = 0.524          # -110 both ways


def recorded_lines():
    """Posted lines, exact market match, last snapshot before first pitch."""
    s = pd.read_parquet(os.path.join(config.PROCESSED_DIR,
                                     "prop_snapshots.parquet"))
    keep = s[s["prop_type"].isin(MARKETS)].copy()
    return (keep.sort_values("snapshot_utc")
            .drop_duplicates(["game_pk", "espn_athlete_id", "prop_type"],
                             keep="last"))


def actuals(pa, season):
    """What actually happened, in the same units the markets are posted in."""
    d = pa[pa["is_pa"] & (pa["season"] == season)].copy()
    w = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
    d["tb"] = d["outcome"].map(w).fillna(0)
    d["hit"] = d["outcome"].isin(["1B", "2B", "3B", "HR"]).astype(int)

    bat = d.groupby(["game_pk", "batter"]).agg(
        act_tb=("tb", "sum"), act_hits=("hit", "sum")).reset_index()
    # Hits+runs+RBI needs run attribution, which props already reconstructs.
    hrr = props.actual_batter_stats(pa, season)
    keep = [c for c in ("game_pk", "batter", "hrr") if c in hrr.columns]
    if "hrr" in hrr.columns:
        bat = bat.merge(hrr[keep], on=["game_pk", "batter"], how="left")
        bat = bat.rename(columns={"hrr": "act_hrr"})

    st = d[d["is_starter"]]
    pit = st.groupby(["game_pk", "pitcher"]).agg(
        act_k=("outcome", lambda s: (s == "K").sum()),
        act_outs=("outs_pa", "sum")).reset_index()
    return bat, pit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()

    lines = recorded_lines()
    print(f"{len(lines):,} posted lines across {lines['game_pk'].nunique()} "
          f"games, in {lines['prop_type'].nunique()} markets")

    pa = om.load_pa()
    want = set(int(g) for g in lines["game_pk"].unique())
    games = [g for g in evaluate_sim.extract_games(pa, args.season)
             if int(g["game_pk"]) in want]
    print(f"{len(games)} of those games are played and in the PA data\n")
    if not games:
        raise SystemExit("no overlap between recorded lines and played games")

    ps = props.PropSimulator(pa, args.season)
    rng = np.random.default_rng(19)

    rows = []
    for g in games:
        d = ps.run(g, args.sims, rng)
        for ti, side in enumerate(("away", "home")):
            rows.append({"game_pk": int(g["game_pk"]),
                         "player": int(g[f"{side}_starter"]), "kind": "pitcher",
                         "starter_k": d["starter_k"][:, ti],
                         "starter_outs": d["starter_outs"][:, ti]})
            for slot, bid in enumerate(g[f"{side}_lineup"]):
                rows.append({
                    "game_pk": int(g["game_pk"]), "player": int(bid),
                    "kind": "hitter",
                    "bat_total_bases": d["bat_total_bases"][:, ti, slot],
                    "bat_hits": d["bat_hits"][:, ti, slot],
                    "bat_hrr": d["bat_hrr"][:, ti, slot]})
    proj = pd.DataFrame(rows)

    names = lp.name_lookup(sorted(set(int(p) for p in proj["player"])))
    proj["nm"] = proj["player"].map(lambda p: lp.normalize(names.get(int(p), "")))
    lines["nm"] = lines["player"].map(lp.normalize)

    bat_act, pit_act = actuals(pa, args.season)

    print(f"{'market':<16}{'n':>5}{'line':>7}{'mean':>8}{'P(over)':>8}"
          f"{'unders':>8}{'won':>10}")
    print("-" * 62)
    for market, (kind, key, label) in MARKETS.items():
        ml = lines[lines["prop_type"] == market]
        if not len(ml):
            continue
        # `lines` already has a `player` column holding the ESPN name string,
        # so the id column is renamed before merging or pandas suffixes both
        # into player_x/player_y and the later join loses its key.
        sub = proj[proj["kind"] == kind][["game_pk", "player", "nm", key]]
        sub = sub.rename(columns={"player": "pid"})
        j = ml.merge(sub, on=["game_pk", "nm"], how="inner")
        avail = ml[ml["game_pk"].isin({int(g["game_pk"]) for g in games})]
        if not len(j):
            print(f"{label:<16}   no name matches ({len(avail)} available)")
            continue

        act = bat_act if kind == "hitter" else pit_act
        idcol = "batter" if kind == "hitter" else "pitcher"
        acol = {"bat_total_bases": "act_tb", "bat_hits": "act_hits",
                "bat_hrr": "act_hrr", "starter_k": "act_k",
                "starter_outs": "act_outs"}[key]
        if acol not in act.columns:
            continue
        j = j.merge(act[["game_pk", idcol, acol]].rename(
            columns={idcol: "pid"}), on=["game_pk", "pid"], how="inner")
        j = j.dropna(subset=[acol])
        if not len(j):
            continue

        # Side selection from P(over), not from the mean. Outs recorded is
        # spiky rather than smooth -- managers pull starters between innings,
        # so 20.4% of 2026 starts ended at exactly 18 outs and another peak
        # sits at 15, with the values between them nearly empty. A long left
        # tail of short starts then drags the mean below the median. A pitcher
        # can project 17.64 outs on the mean and still be 63% to clear 17.5,
        # because the line asks a quantile question and the mean can land in a
        # valley where almost no probability lives. Comparing the mean to the
        # line picked the wrong side on exactly those props.
        j["pred"] = j[key].map(np.mean)
        j["p_over"] = [float(np.mean(np.asarray(v) > ln))
                       for v, ln in zip(j[key], j["line"])]
        j["edge"] = j["p_over"] - 0.5
        push = int((j[acol] == j["line"]).sum())
        won = np.where(j["edge"] > 0, j[acol] > j["line"], j[acol] < j["line"])

        # Base-rate benchmark. Comparing a win rate to 52.4% only makes sense
        # when the line is near a coin flip. When the model takes one side
        # almost every time -- 97% unders on total bases -- the right question
        # is whether it beats betting that side blindly, not whether it beats
        # break-even. P(TB > 1.5) is 35.2%, so blind unders win 64.8%, and a
        # 60.0% model is *worse* than naive while showing +4.8 sigma against
        # 52.4%. The benchmark below is the empirical rate of whichever side
        # the model chose, computed per line within this market.
        grp = j.groupby("line")[acol]
        over_rate = j["line"].map(
            grp.apply(lambda x: float((x > x.name).mean())))
        chose_over = j["edge"].to_numpy() > 0
        blind = np.where(chose_over, over_rate.to_numpy(),
                         1.0 - over_rate.to_numpy())
        settled = len(j) - push
        wr = won.sum() / settled if settled else float("nan")
        unders = (j["p_over"] < 0.5).mean() * 100
        print(f"{label:<16}{len(j):>5}{j['line'].mean():>7.2f}"
              f"{j['pred'].mean():>8.2f}{j['p_over'].mean():>8.2f}"
              f"{unders:>7.0f}%{wr * 100:>9.1f}%")
        # Standard error on the win rate, so a number over break-even is read
        # with the sample size attached rather than on its own.
        if settled:
            se = np.sqrt(0.25 / settled) * 100
            bl = float(np.nanmean(blind)) * 100
            print(f"{'':<16}   match {len(j)}/{len(avail)} lines, "
                  f"{push} pushes, SE {se:.1f}pp")
            print(f"{'':<16}   blind base rate for the side taken {bl:.1f}%"
                  f"  ->  {wr * 100 - bl:+.1f}pp "
                  f"({(wr * 100 - bl) / se:+.1f} sigma)")

    print("\nNOTE: this measures the model against the *number*, not the price.")
    print("ESPN's feed does not label which side each price belongs to, so the")
    print("log-loss complementarity test cannot be run on these markets.")


if __name__ == "__main__":
    main()
