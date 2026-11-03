"""Tonight's slate: prop probabilities for games that have not been played.

Everything the backtest does, pointed at a future date. Two differences matter:

* **Projections use every game through yesterday**, not the month-boundary
  buckets the backtest uses. Monthly granularity keeps a backtest honestly
  leak-free; for a live prediction it would throw away everything since the
  first of the month, which on the 31st is the entire month.
* **Lineups come from StatsAPI**, not from the play-by-play of a completed
  game. This is the prediction point the project committed to from the start:
  after lineups post, roughly two to three hours before first pitch. Games
  without a posted lineup are reported as waiting rather than guessed at.

    python live_props.py                    # today
    python live_props.py --date 2026-08-01 --sims 20000
"""

import argparse
import datetime as dt
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import baserunning
import bullpen
import config
import outcome_model as om
import player_ratings as pr
import props
import simulate as sim

# No custom User-Agent, matching record_odds.py and fetch_park_factors.py.
# This is the last copy of the string ESPN began 403ing on 2026-08-04. Only
# StatsAPI is called from here and it accepts anything, so this is hardening --
# but a malformed browser UA is exactly what tripped the spoof check there.
UA = None
STATSAPI = "https://statsapi.mlb.com/api/v1"

# StatsAPI abbreviations differ from Statcast's for a few clubs.
ABBREV = {"AZ": "AZ", "ARI": "AZ", "CWS": "CWS", "CHW": "CWS", "WSH": "WSH",
          "SD": "SD", "SF": "SF", "TB": "TB", "KC": "KC", "OAK": "ATH",
          "ATH": "ATH"}


def fetch_slate(date_str):
    """Games, probable starters, and posted lineups for a date."""
    r = requests.get(f"{STATSAPI}/schedule", params={
        "sportId": 1, "date": date_str,
        "hydrate": "probablePitcher,lineups,team"}, timeout=30, headers=UA)
    dates = r.json().get("dates", [])
    if not dates:
        return []

    out = []
    for g in dates[0]["games"]:
        lu = g.get("lineups", {}) or {}
        away_lu = lu.get("awayPlayers", []) or []
        home_lu = lu.get("homePlayers", []) or []
        ap = (g["teams"]["away"].get("probablePitcher") or {})
        hp = (g["teams"]["home"].get("probablePitcher") or {})
        out.append({
            "game_pk": g["gamePk"],
            "start_utc": pd.to_datetime(g["gameDate"], utc=True),
            "status": g["status"]["detailedState"],
            "away_team": ABBREV.get(g["teams"]["away"]["team"]["abbreviation"],
                                    g["teams"]["away"]["team"]["abbreviation"]),
            "home_team": ABBREV.get(g["teams"]["home"]["team"]["abbreviation"],
                                    g["teams"]["home"]["team"]["abbreviation"]),
            "away_lineup_ids": [p["id"] for p in away_lu],
            "home_lineup_ids": [p["id"] for p in home_lu],
            "away_starter": ap.get("id"),
            "home_starter": hp.get("id"),
            "away_starter_name": ap.get("fullName", "TBD"),
            "home_starter_name": hp.get("fullName", "TBD"),
        })
    return out


def name_lookup(ids):
    """Player names, one call for the whole slate.

    Full names rather than boxscore names, because these are matched against
    the market feed by name -- the two sources share no player id.
    """
    ids = [int(i) for i in ids if i]
    if not ids:
        return {}
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        r = requests.get(f"{STATSAPI}/people",
                         params={"personIds": ",".join(map(str, chunk))},
                         timeout=30, headers=UA)
        for p in r.json().get("people", []):
            out[p["id"]] = p.get("fullName") or p.get("boxscoreName")
    return out


def normalize(name):
    """Strip accents and punctuation so two feeds' spellings agree."""
    import unicodedata
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def american_to_implied(odds):
    """Implied probability from an American price.

    NOT USED for edges, deliberately. ESPN's prop feed posts one row per player
    per prop with no field saying whether the price is the over or the under,
    and it demonstrably mixes them: total-bases rows appear at both +129
    (implying 43.7%, plausible for an over on 1.5) and around -130 (implying
    56.5%, which no hitter reaches for two-plus bases and so must be an under).
    Reading every price as an over produced edges that were uniformly negative
    on one prop type and uniformly positive on another -- the signature of a
    mapping error, since genuine edges scatter in sign.

    The LINE is trustworthy and is what gets compared against. The price is
    recorded for later, in case the side can be resolved from another source.
    """
    if odds is None or not np.isfinite(odds):
        return None
    return (-odds / (-odds + 100.0)) if odds < 0 else (100.0 / (odds + 100.0))


def load_market(date_str):
    """Latest prop snapshot per player and prop type."""
    import record_odds
    import os
    if not os.path.exists(record_odds.PROPS_LOG):
        return None
    d = pd.read_parquet(record_odds.PROPS_LOG)
    d["game_date"] = d["game_date"].astype(str)
    d = d[d["game_date"] == date_str]
    if not len(d):
        return None
    d = d.sort_values("snapshot_utc").drop_duplicates(
        ["game_pk", "player", "prop_type"], keep="last")
    d["key"] = d["player"].map(normalize)
    return d


class LiveSimulator(props.PropSimulator):
    """PropSimulator with a single as-of model built through a given date."""

    def __init__(self, pa, season, asof_date):
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

        # Pinch hitting, matching PropSimulator. Without it a starter keeps
        # batting after he has been lifted, which overstates hitter counting
        # lines by about 3% -- the exact bias the attribution fix removed from
        # the backtest, and it would otherwise be back in every live number.
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

        # Everything played before today, which is the whole point of a live
        # build: a month-boundary bucket would discard the current month.
        done = pa[(pa["season"] == season)
                  & (pd.to_datetime(pa["game_date"]) < pd.Timestamp(asof_date))]
        std = om.build_count_tables(done) if len(done) else None
        self.model = om.OutcomeModel(tables, season, season_to_date=std)
        print(f"projections use {done['game_pk'].nunique():,} games "
              f"through {pd.to_datetime(done['game_date']).max().date()}")

        # A batter's side against each pitcher hand, for switch hitters.
        d = pa[pa["is_pa"] & (pa["season"] == season)]
        self.side = {}
        for (b, h), grp in d.groupby(["batter", "p_throws"]):
            self.side[(int(b), h)] = grp["stand"].mode().iloc[0]

        # Each starter's recent workload, for the hook.
        recent = (pa[pa["is_pa"] & pa["is_starter"]
                     & (pa["season"] == season)]
                  .groupby(["pitcher", "game_pk"])["outs_pa"].sum()
                  .groupby("pitcher").mean())
        self.recent_outs = recent.to_dict()

    def bat_side(self, pid, vs_hand):
        return self.side.get((int(pid), vs_hand), "R")

    def run_live(self, game, n_sims, rng):
        away = np.array(game["away_lineup_ids"], dtype=int)
        home = np.array(game["home_lineup_ids"], dtype=int)
        a_hand = self.hands.get(int(game["away_starter"]), "R")
        h_hand = self.hands.get(int(game["home_starter"]), "R")

        away_stands = np.array([self.bat_side(p, h_hand) for p in away])
        home_stands = np.array([self.bat_side(p, a_hand) for p in home])

        pen_arms, pen_hand = [], np.zeros((2, sim.MAX_PITCHERS), dtype=np.int8)
        today = game["start_utc"].tz_convert(None)
        for ti, team in enumerate((game["away_team"], game["home_team"])):
            ids, w, avail, entry = self.rosters.as_of(team, today,
                                                      with_entry=True)
            seq = bullpen.bullpen_sequence(ids, w, entry, available=avail)
            sh = [self.hands.get(int(p), "R") for p in seq]
            pen_arms.append((seq, sh))
            for k, hd in enumerate(sh[:sim.MAX_PITCHERS - 1]):
                pen_hand[ti, k + 1] = 1 if hd == "L" else 0

        cum = sim.build_cum_probs(
            self.model, away, home, away_stands, home_stands,
            int(game["away_starter"]), int(game["home_starter"]),
            a_hand, h_hand, self.reliever_vec,
            park_mult=self.park_factors.get(game["home_team"]),
            calibration=self.calib, home_advantage=self.home_adv,
            bullpen_arms=pen_arms, pinch_vec=self.pinch_vec)

        lineup_speed = np.array(
            [[self.speed_ratings.get(int(p)) for p in away],
             [self.speed_ratings.get(int(p)) for p in home]], dtype=np.float32)
        lineup_side = np.array(
            [[1 if s == "L" else 0 for s in away_stands],
             [1 if s == "L" else 0 for s in home_stands]], dtype=np.int8)

        eo = [float(self.recent_outs.get(int(game["away_starter"]), 15.5)),
              float(self.recent_outs.get(int(game["home_starter"]), 15.5))]

        away_sc, home_sc, _, diag = sim.simulate(
            cum, self.table, n_sims, rng, tto_mult=self.tto_mult,
            lineup_speed=lineup_speed, hook=self.hook,
            starter_expected_outs=eo, starter_is_opener=[0.0, 0.0],
            dispersion_sigma=0.19, pen_hand=pen_hand,
            lineup_side=lineup_side, diagnostics=True, pinch=self.pinch)
        # Final scores are returned so a caller can resolve a pitcher win.
        # Daily fantasy pays 4 points for one and it cannot be recovered from
        # the diagnostics alone -- omitting it cost every simulated starter
        # about 1.5 points against Yahoo's own projection.
        diag["final_score"] = np.stack([away_sc, home_sc], axis=1)
        return diag, (away, home)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    slate = fetch_slate(args.date)
    ready = [g for g in slate
             if len(g["away_lineup_ids"]) == 9 and len(g["home_lineup_ids"]) == 9
             and g["away_starter"] and g["home_starter"]
             and g["status"] in ("Pre-Game", "Scheduled", "Warmup")]
    print(f"{args.date}: {len(slate)} games, {len(ready)} ready "
          f"(lineups posted and not yet started)\n")
    for g in slate:
        if g not in ready:
            why = ("started" if g["status"] not in
                   ("Pre-Game", "Scheduled", "Warmup") else "no lineup yet")
            print(f"  skipping {g['away_team']} @ {g['home_team']} "
                  f"({g['start_utc']:%H:%M}Z) -- {why}")
    if not ready:
        print("\nnothing to price. Lineups post 2-3 hours before first pitch.")
        return

    # Refresh rosters before pricing. run_props.bat already does this, but
    # pricing must not silently use yesterday's roster when live_props is run
    # on its own -- IL moves land the morning of a game, so a one-day-old
    # snapshot can put an unavailable player in a simulated bullpen or bench.
    import active_rosters
    active_rosters.ensure_today()

    pa = om.load_pa()
    ls = LiveSimulator(pa, 2026, args.date)
    names = name_lookup(
        [g[k] for g in ready for k in ("away_starter", "home_starter")]
        + [p for g in ready for p in g["away_lineup_ids"] + g["home_lineup_ids"]])

    market = load_market(args.date)
    if market is not None:
        print(f"market: {len(market):,} prop lines recorded for {args.date}")
    else:
        print("market: no prop lines recorded yet "
              "(run record_odds.py to capture them)")

    out_rows = []

    pick_rows = []
    lines_txt = []

    def emit(s=""):
        print(s)
        lines_txt.append(s)

    def market_row(gpk, player, prop_type):
        if market is None or not player:
            return None
        m = market[(market["game_pk"] == gpk)
                   & (market["key"] == normalize(player))
                   & (market["prop_type"] == prop_type)]
        return m.iloc[0] if len(m) else None

    for g in ready:
        diag, (away, home) = ls.run_live(g, args.sims, rng)
        emit(f"\n{'=' * 74}")
        emit(f"{g['away_team']} @ {g['home_team']}   "
             f"first pitch {g['start_utc']:%H:%M}Z")
        emit("=" * 74)

        for ti, (spid, team) in enumerate(
                ((g["away_starter"], g["away_team"]),
                 (g["home_starter"], g["home_team"]))):
            k = diag["starter_k"][:, ti]
            outs = diag["starter_outs"][:, ti]
            nm = names.get(int(spid), str(spid))
            # Outs, not innings: the market posts outs recorded, so printing
            # innings forces a conversion at exactly the moment a line is being
            # read. Median alongside the mean because the distribution is spiky
            # -- starters come out between innings, so the mass piles at
            # multiples of three and the mean can sit in an empty valley
            # between them, below the median. Comparing the mean to a 17.5 line
            # is misleading; the P(over) figures below are the ones to use.
            emit(f"\n  {nm} ({team})  proj {k.mean():.2f} K, "
                 f"{outs.mean():.1f} outs (median {np.median(outs):.0f}, "
                 f"{outs.mean() / 3:.2f} IP)")

            mk = market_row(g["game_pk"], nm, "Total Strikeouts")
            if mk is not None and mk["line"] is not None:
                ln = float(mk["line"])
                p_model = float((k > ln).mean())
                lean = "OVER" if p_model > 0.5 else "under"
                emit(f"    line {ln}   model over {p_model * 100:5.1f}%   "
                     f"lean {lean}   (proj {k.mean():.2f} vs line {ln})")
                out_rows.append({
                    "date": args.date, "game_pk": g["game_pk"], "player": nm,
                    "team": team, "prop": "Total Strikeouts", "line": ln,
                    "model_prob_over": p_model, "model_mean": float(k.mean()),
                    "mean_minus_line": float(k.mean()) - ln,
                    "american_side_unknown": mk["american"]})
                # Recorded before first pitch, so the record cannot be tidied
                # up afterwards. p_side is the probability of the side taken,
                # which is what the confidence split reads.
                pick_rows.append({
                    "game_pk": int(g["game_pk"]), "game_date": args.date,
                    "player": int(spid), "market": "strikeouts", "line": ln,
                    "side": "over" if p_model > 0.5 else "under",
                    "p_side": max(p_model, 1.0 - p_model)})
            else:
                emit("    K over:  " + "   ".join(
                    f"{ln}: {(k > ln).mean() * 100:4.1f}%"
                    for ln in (3.5, 4.5, 5.5, 6.5, 7.5)))
            emit(f"    outs over 14.5: {(outs > 14.5).mean() * 100:4.1f}%   "
                 f"17.5: {(outs > 17.5).mean() * 100:4.1f}%")

        emit("\n  hitters (projected, with market where posted):")
        rows = []
        for ti, lineup in ((0, away), (1, home)):
            tm = g["away_team"] if ti == 0 else g["home_team"]
            for slot in range(9):
                pid = int(lineup[slot])
                nm = names.get(pid, str(pid))
                tb = diag["bat_total_bases"][:, ti, slot]
                hrr = diag["bat_hrr"][:, ti, slot]
                rows.append((float(tb.mean()), nm, tm, tb, hrr,
                             g["game_pk"]))

        for mean_tb, nm, tm, tb, hrr, gpk in sorted(rows, reverse=True)[:6]:
            parts = [f"    {nm:<22} {tm:<4} {mean_tb:.2f} TB"]
            for prop, arr in (("Total Hits + Runs + RBIs", hrr),
                              ("Total Bases", tb)):
                mk = market_row(gpk, nm, prop)
                if mk is None or mk["line"] is None:
                    continue
                ln = float(mk["line"])
                p_model = float((arr > ln).mean())
                parts.append(f"{prop.split()[-1]} o{ln}: "
                             f"{p_model * 100:4.1f}%")
                out_rows.append({
                    "date": args.date, "game_pk": gpk, "player": nm,
                    "team": tm, "prop": prop, "line": ln,
                    "model_prob_over": p_model,
                    "model_mean": float(arr.mean()),
                    "mean_minus_line": float(arr.mean()) - ln,
                    "american_side_unknown": mk["american"]})
            emit("   ".join(parts))

    # --- write results out ------------------------------------------------
    # Ledger: record tonight's picks, grade whatever has settled, and show
    # the running record. Grading happens on every run, so a pick made days ago
    # is scored as soon as its game appears in the PA data.
    try:
        import pick_ledger
        if pick_rows:
            pick_ledger.record(pick_rows)
        pick_ledger.report(emit=emit)
    except Exception as exc:      # never let bookkeeping break pricing
        emit(f"  (pick ledger unavailable: {type(exc).__name__}: {exc})")

    import os
    out_dir = os.path.join(config.PROJECT_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%H%M")
    txt_path = os.path.join(out_dir, f"props_{args.date}_{stamp}.txt")
    csv_path = os.path.join(out_dir, f"props_{args.date}_{stamp}.csv")

    header = (f"MLB prop projections\n"
              f"date {args.date}   generated {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n"
              f"{len(ready)} of {len(slate)} games priced "
              f"(the rest had no lineup posted yet)\n"
              f"{args.sims:,} simulations per game\n\n"
              "Market LINES are shown; market PRICES are not. ESPN's prop feed\n"
              "posts one row per player per prop without saying whether the\n"
              "price is the over or the under, and it mixes them -- so an\n"
              "implied probability computed from it is meaningless. The line\n"
              "itself is reliable, and the model's probability of clearing it\n"
              "is what is reported.\n\n"
              "Pitcher strikeout lines calibrated within ~1 point in backtest\n"
              "(3.5: 67.5%/68.8%, 4.5: 51.0%/50.7%, 5.5: 34.8%/35.3%).\n"
              "Hitter props are much weaker -- total-bases correlation +0.13,\n"
              "barely better than predicting the mean. Treat them as colour.\n")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n".join(lines_txt) + "\n")

    if out_rows:
        df = pd.DataFrame(out_rows)
        df = df.sort_values("mean_minus_line", ascending=False)
        df.to_csv(csv_path, index=False)
        print(f"\nwrote {txt_path}")
        print(f"wrote {csv_path}  ({len(df)} props with a market line)")

        k = df[df["prop"] == "Total Strikeouts"].copy()
        if len(k):
            print("\nstarter strikeout leans (the calibrated market):")
            for r in k.itertuples(index=False):
                lean = "OVER " if r.model_prob_over > 0.5 else "under"
                print(f"  {r.player:<22} line {r.line:<5} "
                      f"proj {r.model_mean:5.2f}  "
                      f"over {r.model_prob_over * 100:5.1f}%  {lean}")
    else:
        print(f"\nwrote {txt_path} (no market lines matched)")


if __name__ == "__main__":
    main()
