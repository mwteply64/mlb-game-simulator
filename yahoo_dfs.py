"""Yahoo daily fantasy: pull the slate, project it, and build lineups.

Three parts, deliberately separable so each can be checked on its own.

**The feed.** Yahoo's DFS API is open and unauthenticated. `contestsFilteredWeb`
lists contests with salary caps; `contestPlayers` returns the pool with salaries,
eligible positions, confirmed batting order, and Yahoo's own `projectedPoints`.
The series payload also carries a per-game weather forecast -- wind speed and
direction, temperature, precipitation -- which is the single largest contextual
factor in scoring and which this project had no other source for.

Yahoo's own projections are worth as much as the salaries: they give a free,
continuously updated benchmark. If our numbers cannot beat Yahoo's on their own
slate, that is worth discovering before entering anything, not after.

**The scoring.** Recovered from Yahoo's data rather than copied from a rules
page, so it is verifiable. Every hitter game-score in the feed is an exact
non-negative integer combination of {2.6, 1.9, 4.2} -- 400 of 400 tested -- which
pins hitters at 2.6 per total base, 2.6 per walk or hit-by-pitch, 1.9 per run,
1.9 per RBI and 4.2 per stolen base.

Pitcher scoring could NOT be recovered the same way and is taken from Yahoo's
published rules instead: win 4, strikeout 2, out 1, earned run -2, and -0.9 each
for a hit, walk or hit batsman. Recovery failed because `fantasyPointsHistory`
holds the last ten *team* games rather than the last ten games played -- a
starter's entries are mostly days he did not pitch -- so there was nothing
reliable to align starts against. Five plausible scoring systems each reproduced
only 5.6-9.6% of listed scores; the published values reproduce 60.9% with a
median difference of exactly 0.00, which is the confirmation.

One known bias: the -2 applies to *earned* runs and the simulator does not
separate earned from unearned, so pitchers are over-penalised by roughly 0.07
points a start. Small, but systematic and against pitchers in every lineup.

**The optimizer.** An exact integer program over the whole board at once
(`scipy.optimize.milp`, HiGHS branch-and-bound), never a greedy position-by-
position fill. Greedy filling is the classic way lineups get spoiled: by the
time it reaches the last slot the salary is gone, so it punts catcher whether or
not the board's best remaining value was a catcher. Here every position competes
for every dollar simultaneously, and a cheap player appears only when spending
elsewhere genuinely buys more points.

Verified against Yahoo's official rules page and its own CSV export on
2026-08-22: the roster is P P C 1B 2B 3B SS OF OF OF against a 200 cap, both
scoring tables match exactly, and all 573 salaries and positions agree with the
export. Every player is eligible at exactly ONE position -- the assignment
variable per (player, slot) is therefore inert today, kept only because it costs
nothing if that ever changes.

Single-game contests use a different roster entirely (MEGASTAR at 2x, SUPERSTAR
at 1.5x, three FLEX) and this module does not handle them.

    python yahoo_dfs.py --list
    python yahoo_dfs.py --contest 15742859 --lineups 5
"""

import argparse
import json
import os
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import config
import field_sim

BASE = "https://dfyql-ro.sports.yahoo.com/v2"

# Recovered from the feed, not copied. See module docstring.
HIT_PER_BASE = 2.6      # 1B 2.6, 2B 5.2, 3B 7.8, HR 10.4
HIT_PER_BB = 2.6        # walks and hit-by-pitch alike
HIT_PER_RUN = 1.9
HIT_PER_RBI = 1.9
HIT_PER_SB = 4.2

PITCHER_SCORING_PATH = os.path.join(config.PROCESSED_DIR,
                                    "yahoo_pitcher_scoring.json")

# Yahoo MLB roster: two pitchers, not one, then C/1B/2B/3B/SS and three
# outfielders -- ten players against a 200 salary cap. The API does not publish
# this, so it is set from the lineup builder; a wrong slot count produces
# illegal lineups silently. Two pitcher slots also make the pitcher scoring
# coefficients matter more, since they carry two of ten spots and the largest
# salaries on the board.
ROSTER = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}


def get(path, **params):
    r = requests.get(f"{BASE}/{path}", params=params or None, timeout=30)
    r.raise_for_status()
    return r.json()


def contests(sport="mlb", upcoming_only=True):
    j = get("contestsFilteredWeb", sport=sport)
    res = j["contests"]["result"]
    if upcoming_only:
        res = [c for c in res if c.get("state") == "upcoming"]
    return pd.DataFrame(res)


def slate(contest_id):
    """Player pool plus the contest's salary cap and per-game weather."""
    j = get(f"contestPlayers", contestId=contest_id)
    pl = j["players"]
    pl = pl["result"] if isinstance(pl, dict) else pl
    df = pd.DataFrame(pl)
    df["name"] = df["firstName"].str.cat(df["lastName"], sep=" ")
    df["game_code"] = df["playerGameCode"].str.split("$").str[-1]

    # Opponent club, from the two teams sharing a game code. Needed so the
    # optimizer can refuse to roster a pitcher against its own hitters.
    opp = {}
    for _, g in df.groupby("game_code"):
        ts = [t for t in g["teamAbbr"].dropna().unique()]
        if len(ts) == 2:
            opp[ts[0]], opp[ts[1]] = ts[1], ts[0]
    df["opponent"] = df["teamAbbr"].map(opp)

    d = get(f"contest/{contest_id}")
    cc = d["contests"]["result"][0]
    cap = cc.get("salaryCap")
    weather = {}
    for g in (cc.get("series") or {}).get("games", []):
        weather[g["gameCode"]] = g.get("forecast") or {}
    return df, cap, weather


def pick_slate(sport="mlb", when=None, prefer="most_games"):
    """Choose which slate to build for.

    Yahoo runs several slates a day -- an all-day card, an evening card, express
    slates of two or three games -- and they are distinguished by `seriesId`,
    not by contest. Contests sharing a seriesId share a player pool and lock
    time, so the slate is the real unit and the contest only decides the fee and
    payout shape.

    Defaults to the upcoming slate with the most games, which is normally the
    main card and the one with enough player pool for the optimizer to have
    choices. Pass `when` as a date string to restrict to a day.
    """
    c = contests(sport)
    if not len(c):
        return None
    c = c[c["state"] == "upcoming"].copy()
    if when:
        day = pd.Timestamp(when).date()
        c = c[pd.to_datetime(c["startTime"], unit="ms").dt.date == day]
    if not len(c):
        return None

    out = []
    for sid, g in c.groupby("seriesId"):
        d = get(f"contest/{int(g.iloc[0]['id'])}")
        series = (d["contests"]["result"][0].get("series") or {})
        out.append({
            "seriesId": sid,
            "name": series.get("name"),
            "games": len(series.get("games") or []),
            "start": pd.to_datetime(series.get("startTime"), unit="ms"),
            "contests": len(g),
            "example_contest": int(g.iloc[0]["id"]),
            "salaryCap": g.iloc[0]["salaryCap"],
        })
    df = pd.DataFrame(out).sort_values(
        "games" if prefer == "most_games" else "start", ascending=False)
    return df.reset_index(drop=True)


def eligible_pool(df):
    """Drop only the injured and the confirmed non-starters.

    Yahoo's `starting` field has three values, not two: "true" for a confirmed
    starter, "false" for a player whose team HAS posted a lineup that omits him,
    and "unknown" for a team that has not posted yet. Filtering on
    `starting == "true"` would silently discard every player on every team
    awaiting a lineup -- 78 players on this slate -- which is both a large loss
    of pool and biased toward whichever clubs post early.

    So "unknown" is kept. The cost is that some of those players will end up on
    the bench; the alternative is not considering entire teams.
    """
    d = df.copy()
    status = d["status"].fillna("N/A").astype(str)
    start = d["starting"].fillna("unknown").astype(str)

    on_il = status.str.upper().str.startswith("IL")
    # Day-to-day is only disqualifying while unconfirmed; if he is in a posted
    # lineup he is playing, whatever the tag says.
    dtd_out = (status.str.upper() == "DTD") & (start != "true")
    benched = start == "false"

    keep = ~(on_il | dtd_out | benched)
    d = d[keep].copy()
    d["lineup_confirmed"] = start[keep] == "true"
    return d


def hitter_points(d, ti, slot):
    """Yahoo points for one lineup slot, from simulated components."""
    tb = (d["bat_1b"][:, ti, slot] + 2 * d["bat_2b"][:, ti, slot]
          + 3 * d["bat_3b"][:, ti, slot] + 4 * d["bat_hr"][:, ti, slot])
    return (HIT_PER_BASE * tb
            + HIT_PER_BB * d["bat_bb"][:, ti, slot]
            + HIT_PER_RUN * d["bat_r"][:, ti, slot]
            + HIT_PER_RBI * d["bat_rbi"][:, ti, slot])


def points_per_start(pa, season):
    """Yahoo points per game, counting only games the player STARTED.

    Yahoo's `fantasyPointsPerGame` averages over every game a player appeared
    in, including ones he entered late for a single plate appearance. For a
    part-time player that denominator is badly depressed -- Jahmai Jones started
    28 of 70 games and averaged 1.4 PA in the other 42 -- so comparing a
    full-game projection against it inflates the ratio and flags exactly the
    players it is least able to judge.

    Restricting to started games makes the comparison like-for-like: what this
    player produces when he plays a full game, against what we project for one.

    Stolen bases are omitted (4.2 each); the PA table does not carry them, and
    they are a small share of hitter scoring.
    """
    import props as _props
    d = pa[pa["is_pa"] & (pa["season"] == season)]
    # A slot's first plate appearance belongs to whoever started there.
    first = (d.sort_values("at_bat_number")
             .drop_duplicates(["game_pk", "batting_team", "lineup_slot"],
                              keep="first"))
    started = set(zip(first["game_pk"].astype(int),
                      first["batter"].astype(int)))

    st = _props.actual_batter_stats(pa, season)
    for c in ("total_bases", "BB", "HBP", "runs", "rbi"):
        if c not in st.columns:
            st[c] = 0
    st["pts"] = (HIT_PER_BASE * st["total_bases"]
                 + HIT_PER_BB * (st["BB"] + st["HBP"])
                 + HIT_PER_RUN * st["runs"] + HIT_PER_RBI * st["rbi"])
    st["started"] = [(int(g), int(b)) in started
                     for g, b in zip(st["game_pk"], st["batter"])]
    s = st[st["started"]]
    if not len(s):
        return {}, {}
    tot = s.groupby("batter")["pts"].sum()
    n = s.groupby("batter")["pts"].size()
    league = float(s["pts"].mean())

    # Shrink toward the league starting average. A raw mean breaks on thin
    # history in the worst way: Corey Julks had one start and scored zero, so
    # his denominator was 0.00 and the ratio came out NaN -- reading as
    # "unflagged" for a player with essentially no starting record, while he sat
    # in a built lineup. Five starts of prior keeps every denominator finite and
    # pulls short histories toward the league rate rather than trusting them.
    K = 5.0
    per = ((tot + K * league) / (n + K))
    return per.to_dict(), n.to_dict()


def lineup_distribution(samples, player_ids, n_draws=20000, seed=3):
    """Score distribution for a lineup, preserving within-game correlation.

    The projection the optimizer maximises is a *mean*. A winning tournament
    score is the extreme right tail across thousands of entries, so the two are
    different quantities and a system whose mean matched winning scores would be
    badly miscalibrated. What matters for a large-field contest is the ceiling,
    and that requires a distribution.

    `samples` maps player -> (game_key, array of simulated Yahoo points). Draws
    share a simulation index within a game and draw independently across games,
    which is what keeps teammates correlated -- when the leadoff man scores, the
    cleanup hitter's RBI arrives in the *same* simulated game. Summing
    independent per-player distributions would understate the spread badly,
    because it would throw away exactly the covariance that produces big team
    scores. That correlation is the one structural advantage this simulator has
    over an ordinary projection sheet.
    """
    rng = np.random.default_rng(seed)
    by_game = {}
    for pid in player_ids:
        got = samples.get(int(pid))
        if got is None:
            continue
        gk, arr = got
        by_game.setdefault(gk, []).append(np.asarray(arr, dtype=float))
    if not by_game:
        return None

    total = np.zeros(n_draws)
    for gk, arrs in by_game.items():
        n = min(len(a) for a in arrs)
        idx = rng.integers(0, n, size=n_draws)
        for a in arrs:
            total += a[idx]
    return total


def build_ceiling_lineups(pool, cap, samples, n_lineups=5, threshold=170.0,
                          n_candidates=250, max_overlap=1, max_per_team=6,
                          alpha=1.0, seed=5, min_teams=3, min_salary=0):
    """Lineups chosen for upside, not expected points.

    In a 360-entry field roughly one lineup in twenty clears 150, so about
    eighteen entries do -- winning means beating those, which puts the target
    near a lineup's own 99th percentile rather than its mean. Maximising the
    mean is right for cash games and wrong here.

    P(total > threshold) is not linear in the selection, so the integer program
    cannot maximise it directly. Two stages instead:

    1. **Generate** many candidates against a linear proxy, `mean + alpha*sd`
       per player, which is solvable and leans toward volatile players.
    2. **Score** each candidate's true ceiling with the correlated simulator and
       keep the best, subject to the overlap rule.

    Stacking is deliberately *not* hard-coded. Correlated teammates raise the
    real ceiling, so stacked candidates score better at stage two and are
    selected on their merits. Imposing a stacking rule would assert what the
    simulator can measure.

    `max_per_team` is 6 because that is Yahoo's rule -- at most six players
    from one club per lineup, pitchers included. An earlier version left it
    unrestricted, reasoning that a cap suppresses the correlation a tournament
    build wants; that was wrong, because the cap is not a modelling choice but
    a constraint the contest enforces. It produced seven-hitter stacks that
    would have been rejected at entry. Six is still a deep stack.
    """
    rng = np.random.default_rng(seed)
    ids = pool["mlbam"]

    # Per-player spread, from the same simulated draws the ceiling uses.
    sd = {}
    for pid, got in samples.items():
        sd[int(pid)] = float(np.std(np.asarray(got[1], dtype=float)))
    pool = pool.copy()
    pool["sd"] = [sd.get(int(i), 0.0) if pd.notna(i) else 0.0 for i in ids]

    # Teams worth stacking, most productive first. Generation picks one per
    # candidate and boosts it, which is what actually puts stacks in the pool.
    teams = [t for t in pool["team"].dropna().unique()]

    cands = []
    seen = set()
    for k in range(n_candidates):
        p2 = pool.copy()
        # Jitter both terms so successive solves explore, rather than returning
        # the same argmax repeatedly.
        p2["proj"] = (pool["proj"] + alpha * pool["sd"]) * (
            1.0 + rng.normal(0, 0.10, len(pool)))

        # Bias most candidates toward one randomly chosen club. The generation
        # objective is per-player and therefore correlation-blind, so stacked
        # lineups would otherwise enter the pool only by accident -- the first
        # ceiling build produced stacks of two to four and used five or six
        # clubs per lineup, which is not a stack. The scoring stage still
        # decides on the true correlated ceiling; this only ensures stacked
        # candidates exist to be scored. A quarter are left unstacked so a
        # balanced lineup can still win on merit.
        if teams and k % 4:
            t = teams[rng.integers(0, len(teams))]
            boost = (p2["team"] == t) & (p2["primaryPosition"] != "P")
            p2.loc[boost, "proj"] = p2.loc[boost, "proj"] * 1.35
        lus = build_lineups(p2, cap, n_lineups=1, max_per_team=max_per_team,
                            randomness=0.0, min_teams=min_teams,
                            min_salary=min_salary)
        if not lus:
            continue
        key = tuple(sorted(lus[0]["name"]))
        if key in seen:
            continue
        seen.add(key)
        cands.append(lus[0])

    name_to_id = dict(zip(pool["name"], pool["mlbam"]))
    # True per-player projection, to restore after the search. The candidates
    # carry the jittered `mean + alpha*sd` proxy in their `proj` column, which
    # is a search heuristic and roughly double a real projection. Leaving it in
    # place wrote lineups claiming 208 points when the true mean was 107.
    true_proj = dict(zip(pool["name"], pool["proj"]))
    scored = []
    for lu in cands:
        lu["proj"] = [true_proj.get(n, float("nan")) for n in lu["name"]]
        pids = [name_to_id.get(n) for n in lu["name"]]
        pids = [int(x) for x in pids if pd.notna(x)]
        tot = lineup_distribution(samples, pids, n_draws=6000)
        if tot is None:
            continue
        # True mean restored for reporting: the proxy above is a search
        # heuristic, never the number shown.
        real_mean = float(np.mean(tot))
        scored.append((float((tot > threshold).mean()), real_mean,
                       float(np.percentile(tot, 99)), lu))
    scored.sort(key=lambda r: -r[0])

    out = []
    for pwin, mean, p99, lu in scored:
        names = set(lu["name"])
        if any(len(names & set(k["name"])) > max_overlap for k in out):
            continue
        lu = lu.copy()
        lu.attrs["p_over"] = pwin
        lu.attrs["mean"] = mean
        lu.attrs["p99"] = p99
        out.append(lu)
        if len(out) >= n_lineups:
            break
    return out, len(cands)


VEGAS_WEIGHT = 0.9        # fitted: best blend puts 0.9 on the market
POINTS_PER_RUN_EXP = 0.575   # fitted: team hitter points ~ runs ** 0.575


def vegas_adjustment(game_pk, team, sim_runs, implied):
    """Multiplier on a club's hitter projections, from the market's total.

    Returns 1.0 when no total was recorded for the game, so a missing line
    leaves the projection untouched rather than silently zeroing it.
    """
    if sim_runs is None or sim_runs <= 0:
        return 1.0
    row = implied.get((int(game_pk), str(team)))
    if row is None or not np.isfinite(row) or row <= 0:
        return 1.0
    target = VEGAS_WEIGHT * row + (1.0 - VEGAS_WEIGHT) * sim_runs
    return float((target / sim_runs) ** POINTS_PER_RUN_EXP)


def load_pitcher_scoring():
    if os.path.exists(PITCHER_SCORING_PATH):
        with open(PITCHER_SCORING_PATH) as fh:
            return json.load(fh)
    return None


def pitcher_points(d, ti, coef):
    """Yahoo points for a starter, from simulated components.

    The win is worth 4 and has to be resolved from the final score, not the
    diagnostics: a starter is credited when his club wins and he completed five
    innings. Leaving it out understated every simulated pitcher by roughly 1.5
    points -- visible as a 2.6-point gap against Yahoo's own projections, which
    is what exposed it.

    Hit batsmen are counted with walks at -0.9. `starter_bb` holds walks only,
    so `starter_hbp` is added separately rather than folded in, which would have
    changed what the prop output means by that name.
    """
    hbp = d.get("starter_hbp")
    hbp = hbp[:, ti] if hbp is not None else 0.0
    pts = (coef["intercept"]
           + coef["out"] * d["starter_outs"][:, ti]
           + coef["k"] * d["starter_k"][:, ti]
           + coef["hit"] * d["starter_hits"][:, ti]
           + coef["bb"] * d["starter_bb"][:, ti]
           + coef.get("hbp", coef["bb"]) * hbp
           + coef["run"] * coef.get("earned_fraction", 1.0)
           * d["starter_runs"][:, ti])

    score = d.get("final_score")
    if score is not None and coef.get("win"):
        mine, theirs = score[:, ti], score[:, 1 - ti]
        win = (mine > theirs) & (d["starter_outs"][:, ti] >= 15)
        pts = pts + coef["win"] * win
    return pts


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def build_lineups(pool, cap, n_lineups=1, roster=None, max_per_team=None,
                  max_overlap=None, randomness=0.0, seed=0, min_teams=3,
                  min_salary=0):
    """Exact integer program, solved once per lineup.

    `pool` needs: name, salary, proj, positions (list), team.

    Variables are one binary per (player, slot) rather than per player, so a
    player eligible at two positions is genuinely considered at both and the
    solver picks the assignment that frees the most salary elsewhere. Collapsing
    multi-eligible players to a primary position -- the usual shortcut -- throws
    away exactly the flexibility that makes those players valuable.

    `max_overlap` caps how many players a new lineup may share with any earlier
    one, which produces real diversification. Re-running a greedy fill with
    jitter produces near-duplicates instead, because the same few value plays
    dominate every pass.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    roster = roster or ROSTER
    slots = [p for p, n in roster.items() for _ in range(n)]
    slot_names = sorted(roster)
    n_players = len(pool)
    rng = np.random.default_rng(seed)

    # index (i, p) -> column
    cols, idx = [], {}
    for i in range(n_players):
        for p in slot_names:
            if p in pool.iloc[i]["positions"]:
                idx[(i, p)] = len(cols)
                cols.append((i, p))
    n_var = len(cols)
    if not n_var:
        return []

    salary = np.array([pool.iloc[i]["salary"] for i, _ in cols], dtype=float)
    base_proj = np.array([pool.iloc[i]["proj"] for i, _ in cols], dtype=float)

    # One binary per club, appended after the player-slot variables. Yahoo
    # requires at least three clubs in a lineup, which is a property of the set
    # of teams used and cannot be written as a bound on the player variables
    # alone -- it needs an indicator that is 0 unless that club is represented.
    team_list = sorted({str(t) for t in pool["team"].dropna().unique()})
    team_col = {t: n_var + k for k, t in enumerate(team_list)}
    n_total = n_var + len(team_list)
    player_team = [str(pool.iloc[i]["team"]) for i, _ in cols]

    def pad(row):
        """Widen a player-space row to the full variable vector."""
        out = np.zeros(n_total)
        out[:n_var] = row
        return out

    lineups = []
    for L in range(n_lineups):
        proj = base_proj.copy()
        if randomness:
            # Jitter the projections, not the fill order. This explores lineups
            # that are near-optimal under projection uncertainty, which is the
            # honest reason to want variety -- the projections are not precise
            # enough for the single argmax to be meaningfully better than its
            # neighbours.
            proj = proj * (1.0 + rng.normal(0, randomness, size=n_var))

        cons = []
        # Each player used at most once, across all his eligible slots.
        for i in range(n_players):
            row = np.zeros(n_var)
            for p in slot_names:
                if (i, p) in idx:
                    row[idx[(i, p)]] = 1
            if row.any():
                cons.append(LinearConstraint(pad(row), 0, 1))
        # Exactly the required number at each position.
        for p in slot_names:
            row = np.zeros(n_var)
            for i in range(n_players):
                if (i, p) in idx:
                    row[idx[(i, p)]] = 1
            cons.append(LinearConstraint(pad(row), roster[p], roster[p]))
        # Salary cap.
        # Cap is an inequality, deliberately: leaving money unspent is correct
        # whenever nothing affordable improves the lineup. `min_salary` adds a
        # floor only if asked, and forcing it can only make the lineup worse --
        # it moves the solver off the optimum it just found.
        cons.append(LinearConstraint(pad(salary), float(min_salary or 0), cap))

        if max_per_team:
            for t in pool["team"].unique():
                row = np.zeros(n_var)
                for k, (i, p) in enumerate(cols):
                    if pool.iloc[i]["team"] == t:
                        row[k] = 1
                cons.append(LinearConstraint(pad(row), 0, max_per_team))

        if max_overlap is not None:
            for prev in lineups:
                names = set(prev["name"])
                row = np.zeros(n_var)
                for k, (i, p) in enumerate(cols):
                    if pool.iloc[i]["name"] in names:
                        row[k] = 1
                cons.append(LinearConstraint(pad(row), 0, max_overlap))

        # A pitcher may not be rostered against hitters we own. If the
        # pitcher variable is 1, the sum of opposing hitters must be 0; if it
        # is 0 the bound is slack. Big-M is the number of hitter slots.
        if "opp" in pool.columns:
            M = float(sum(v for k, v in roster.items() if k != "P"))
            for k, (i, pslot) in enumerate(cols):
                if pslot != "P":
                    continue
                foe = pool.iloc[i].get("opp")
                if not foe or pd.isna(foe):
                    continue
                row = np.zeros(n_total)
                row[k] = M
                for k2, (i2, p2) in enumerate(cols):
                    if p2 != "P" and str(pool.iloc[i2]["team"]) == str(foe):
                        row[k2] = 1.0
                if row[:n_var].any():
                    cons.append(LinearConstraint(row, -np.inf, M))

        if min_teams and len(team_list) >= min_teams:
            # y_t may only be 1 when at least one player from club t is used,
            # so requiring the indicators to sum to min_teams forces that many
            # distinct clubs into the lineup.
            for t in team_list:
                row = np.zeros(n_total)
                row[team_col[t]] = 1.0
                for k in range(n_var):
                    if player_team[k] == t:
                        row[k] = -1.0
                cons.append(LinearConstraint(row, -np.inf, 0))
            row = np.zeros(n_total)
            for t in team_list:
                row[team_col[t]] = 1.0
            cons.append(LinearConstraint(row, min_teams, np.inf))

        obj = np.zeros(n_total)
        obj[:n_var] = proj
        res = milp(c=-obj, constraints=cons,
                   integrality=np.ones(n_total),
                   bounds=Bounds(0, 1))
        if not res.success or res.x is None:
            break
        chosen = [(cols[k][0], cols[k][1]) for k in range(n_var)
                  if res.x[k] > 0.5]
        lu = pd.DataFrame([{
            "slot": p, "name": pool.iloc[i]["name"],
            "team": pool.iloc[i]["team"], "salary": pool.iloc[i]["salary"],
            "proj": pool.iloc[i]["proj"]} for i, p in chosen])
        lu = lu.sort_values("slot").reset_index(drop=True)
        lineups.append(lu)
    return lineups


def project_slate(contest_id, season=2026, n_sims=3000, date=None,
                  verbose=True):
    """Simulate every game on the slate and price every eligible player.

    Games whose lineup has posted are simulated on that lineup. Games without
    one fall back to `projected_lineups.resolve`, which averages weighted recent
    cards -- and every player from such a game is flagged, because a projection
    built on a guess should never be indistinguishable from one built on a fact.
    """
    import datetime as dt
    import active_rosters
    import live_props as lp
    import outcome_model as om
    import projected_lineups as pj

    date = date or dt.date.today().isoformat()
    pool_raw, cap, weather = slate(contest_id)
    pool = eligible_pool(pool_raw)
    if verbose:
        print(f"pool: {len(pool_raw)} listed -> {len(pool)} eligible "
              f"({int((~pool['lineup_confirmed']).sum())} awaiting a lineup)")

    pa = om.load_pa()
    ls = lp.LiveSimulator(pa, season, date)
    rosters = active_rosters.load()
    games = lp.fetch_slate(date)
    coef = load_pitcher_scoring()
    if coef is None:
        raise SystemExit("no Yahoo pitcher scoring stored; see yahoo_dfs.py")

    yahoo_teams = set(pool["teamAbbr"].str.upper())
    games = [g for g in games
             if g["away_team"] in yahoo_teams or g["home_team"] in yahoo_teams]
    if verbose:
        print(f"{len(games)} slate games matched to StatsAPI")

    # Market implied team totals, if any were recorded for these games.
    try:
        import vegas_totals
        _imp = vegas_totals.implied_totals()
        implied = {(int(r.game_pk), str(r.team)): float(r.implied)
                   for r in _imp.itertuples(index=False)}
    except Exception:
        implied = {}
    if verbose:
        print(f"market totals available for {len(implied)} team-games")

    rng = np.random.default_rng(7)
    proj, play_prob, source_by_team, team_of = {}, {}, {}, {}
    samples = {}          # player -> (game_key, simulated points)
    for g in games:
        if not (g["away_starter"] and g["home_starter"]):
            continue
        cards, srcs = {}, {}
        for side in ("away", "home"):
            team = g[f"{side}_team"]
            opp_hand = ls.hands.get(
                int(g[f"{'home' if side == 'away' else 'away'}_starter"]), "R")
            act = (rosters.as_of(team, pd.Timestamp(date))
                   if rosters.available() else None)
            c, src = pj.resolve(pa, team, g[f"{side}_lineup_ids"],
                                vs_hand=opp_hand, asof=date, season=season,
                                active=act)
            if not c:
                break
            cards[side], srcs[side] = c, src
        if len(cards) != 2:
            continue
        source_by_team[g["away_team"]] = srcs["away"]
        source_by_team[g["home_team"]] = srcs["home"]
        for side, ti in (("away", 0), ("home", 1)):
            team_of[int(g[f"{side}_starter"])] = g[f"{side}_team"]
            for c in cards[side]:
                for pid in c["lineup"]:
                    team_of[int(pid)] = g[f"{side}_team"]

        # Average over lineup candidates. A single guessed card would give a
        # projection as confident as a posted one; weighting several spreads
        # the estimate over what the manager has plausibly been doing.
        acc, total_w, last_samples, best_w = {}, 0.0, {}, -1.0
        for ca in cards["away"]:
            for ch in cards["home"]:
                w = ca["weight"] * ch["weight"]
                total_w += w
                gg = dict(g)
                gg["away_lineup_ids"] = ca["lineup"]
                gg["home_lineup_ids"] = ch["lineup"]
                diag, (aw, hm) = ls.run_live(gg, n_sims, rng)
                keep = w > best_w
                if keep:
                    best_w, last_samples = w, {}
                # One multiplier per club, from its simulated runs against the
                # market's implied total for the same game.
                vadj = {}
                for ti, side in ((0, "away"), (1, "home")):
                    sim_runs = float(diag["bat_rbi"][:, ti, :].sum(axis=1).mean())
                    vadj[ti] = vegas_adjustment(g["game_pk"], g[f"{side}_team"],
                                                sim_runs, implied)
                for ti, ids in ((0, aw), (1, hm)):
                    for slot, pid in enumerate(ids):
                        pts = hitter_points(diag, ti, slot) * vadj[ti]
                        e = acc.setdefault(int(pid), [0.0, 0.0])
                        e[0] += w * float(pts.mean()); e[1] += w
                        if keep:
                            last_samples[int(pid)] = pts.copy()
                for ti, side in ((0, "away"), (1, "home")):
                    pid = int(g[f"{side}_starter"])
                    pts = pitcher_points(diag, ti, coef)
                    e = acc.setdefault(pid, [0.0, 0.0])
                    e[0] += w * float(pts.mean()); e[1] += w
                    if keep:
                        last_samples[pid] = np.asarray(pts, dtype=float).copy()
        # E[points] = P(plays) * E[points | plays]. Dividing by the weight of
        # only the candidates a player appears in would give the second factor
        # alone, so a fringe bat who featured in one of three recent cards would
        # carry the same full-game projection as an everyday regular. Dividing
        # by the total candidate weight applies the probability as well: a
        # player in every candidate is unchanged, which is why top-of-order
        # regulars are unaffected, while an irregular is discounted to what he
        # is actually expected to produce.
        # Keep one representative sample array per player, from the highest
        # weighted candidate, so a ceiling can be computed later. Averaging
        # sample arrays across candidates would blur the distribution; the
        # modal lineup is the honest one to describe spread with.
        for pid, arr in last_samples.items():
            samples[int(pid)] = (int(g["game_pk"]), arr)

        for pid, (tot, wt) in acc.items():
            if total_w:
                proj[pid] = tot / total_w
                if wt:
                    play_prob[pid] = wt / total_w

    names = lp.name_lookup(list(proj))
    # Key on name AND team. There are two Max Muncys in the league -- a Dodgers
    # third baseman at 17 and an Athletics third baseman at 11 -- and a
    # name-only dictionary silently gives both Yahoo rows whichever id was
    # written last, so one of them carries the other's projection. Yahoo's own
    # export has the same collision, which is how it surfaced.
    by_name_team = {}
    by_name = {}
    for pid, nm in names.items():
        k = lp.normalize(nm)
        by_name.setdefault(k, pid)
        t = team_of.get(int(pid))
        if t:
            by_name_team[(k, str(t).upper())] = pid

    def _resolve_id(row):
        k = lp.normalize(row["name"])
        t = str(row["teamAbbr"]).upper()
        # Team-qualified first; fall back to name only, which is safe for
        # everyone who is not half of a duplicate pair.
        return by_name_team.get((k, t), by_name.get(k))

    pool = pool.copy()
    pool["mlbam"] = pool.apply(_resolve_id, axis=1)
    pool["proj"] = pool["mlbam"].map(proj)
    pool["source"] = pool["teamAbbr"].str.upper().map(source_by_team)
    pool["play_prob"] = pool["mlbam"].map(play_prob).fillna(1.0)
    # Sanity band against what this player produces in games he STARTS, not
    # Yahoo's per-appearance average. Not an override -- a hitter promoted to
    # leadoff should beat his own rate -- but a projection well above a player's
    # established starting production is worth seeing before it lands in a
    # lineup. `own_start_games` is carried alongside so a ratio built on five
    # starts is not read like one built on eighty.
    pps, npps = points_per_start(pa, season)
    pool["own_start_avg"] = pool["mlbam"].map(pps)
    pool["own_start_games"] = pool["mlbam"].map(npps)
    pool["vs_own_avg"] = pool["proj"] / pool["own_start_avg"].replace(0, np.nan)
    # Yahoo's own figure kept for reference; it is what their projections and
    # the site display are anchored to.
    pool["yahoo_fppg"] = pd.to_numeric(pool.get("fantasyPointsPerGame"),
                                       errors="coerce")
    pool["positions"] = pool["eligiblePositions"].apply(
        lambda v: list(v) if isinstance(v, (list, tuple)) else [v])
    pool["team"] = pool["teamAbbr"].str.upper()
    pool["opp"] = pool["opponent"].astype(str).str.upper() \
        if "opponent" in pool.columns else None
    out = pool.dropna(subset=["proj"]).copy()
    if verbose:
        print(f"priced {len(out)} of {len(pool)} eligible players "
              f"({int((out['source'] == 'projected').sum())} from projected "
              f"lineups)")
    return out, cap, weather, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--slates", action="store_true")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--contest", type=int)
    ap.add_argument("--lineups", type=int, default=1)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--max-per-team", type=int, default=6,
                    help="Yahoo allows at most 6 from one club, "
                         "pitchers included")
    ap.add_argument("--randomness", type=float, default=0.06)
    ap.add_argument("--threshold", type=float, default=170.0)
    ap.add_argument("--candidates", type=int, default=250)
    ap.add_argument("--min-salary", type=int, default=0,
                    help="floor on spend; the cap is already an inequality so "
                         "this only ever constrains, never improves")
    ap.add_argument("--cash-field", type=int, default=24,
                    help="entries in the small contest; 0 to skip that group")
    ap.add_argument("--gpp-field", type=int, default=360,
                    help="entries in the large contest; 0 to skip that group")
    args = ap.parse_args()

    if args.slates:
        print(pick_slate().to_string(index=False))
        return

    if args.auto and not args.contest:
        sl = pick_slate()
        if sl is None or not len(sl):
            raise SystemExit("no upcoming slates")
        args.contest = int(sl.iloc[0]["example_contest"])
        print(f"auto-selected slate {sl.iloc[0]['name']} "
              f"({sl.iloc[0]['games']} games), contest {args.contest}\n")

    if args.contest:
        pool, cap, weather, samples = project_slate(args.contest,
                                                    n_sims=args.sims)

        import datetime as _dt
        out_dir = os.path.join(config.PROJECT_DIR, "predictions")
        os.makedirs(out_dir, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        base = os.path.join(out_dir, f"dfs_{stamp}")
        name_to_id = dict(zip(pool["name"], pool["mlbam"]))
        src = dict(zip(pool["name"], pool["source"]))

        cols = [c for c in ("name", "teamAbbr", "primaryPosition", "salary",
                            "proj", "projectedPoints", "fantasyPointsPerGame",
                            "own_start_avg", "own_start_games", "vs_own_avg",
                            "play_prob", "source", "starting",
                            "lineupOrder") if c in pool.columns]
        pool[cols].sort_values("proj", ascending=False).to_csv(
            f"{base}_pool.csv", index=False)

        # Which groups to build. A field size of 0 skips that group entirely,
        # so a day with no large contest still produces the small-field set.
        groups = []
        if args.cash_field:
            groups.append(("CASH / SMALL FIELD", "cash", args.cash_field))
        if args.gpp_field:
            groups.append(("TOURNAMENT / LARGE FIELD", "gpp", args.gpp_field))
        if not groups:
            groups = [("LINEUPS", "cash", 0)]

        all_rows, report = [], []
        for title, kind, field_n in groups:
            if kind == "gpp":
                # Upside objective: the mean is nearly irrelevant when a few
                # hundred entries are chasing one prize.
                lus, n_seen = build_ceiling_lineups(
                    pool, cap, samples, n_lineups=args.lineups,
                    threshold=args.threshold, n_candidates=args.candidates,
                    max_overlap=args.max_overlap,
                    max_per_team=args.max_per_team,
                    min_salary=args.min_salary)
                note = (f"upside objective, {n_seen} candidates scored "
                        f"against P(total > {args.threshold:.0f})")
            else:
                # Expected points: in a small field you need to beat roughly
                # half the room, and variance mostly hurts.
                lus = build_lineups(
                    pool, cap, n_lineups=args.lineups,
                    max_per_team=args.max_per_team,
                    max_overlap=args.max_overlap,
                    randomness=args.randomness,
                    min_salary=args.min_salary)
                note = "expected-points objective"

            # Score every lineup, then rank within the group by what that
            # group is actually optimising.
            scored = []
            field = []
            if field_n:
                frng = np.random.default_rng(21)
                field = field_sim.sample_field(
                    pool, cap, field_n, ROSTER, frng,
                    max_per_team=args.max_per_team)
            for lu in lus:
                ids = [name_to_id.get(n) for n in lu["name"]]
                ids = [int(x) for x in ids if pd.notna(x)]
                tot = lineup_distribution(samples, ids)
                if tot is None:
                    continue
                rec = {"lu": lu, "mean": float(tot.mean()),
                       "sd": float(tot.std()),
                       "p99": float(np.percentile(tot, 99)),
                       "p_thresh": float((tot > args.threshold).mean()),
                       "finish": None}
                if field:
                    got = field_sim.finish_distribution(ids, field, samples)
                    if got is not None:
                        place, n_opp = got
                        rec["finish"] = {
                            "n": n_opp + 1,
                            "median": int(np.median(place)),
                            "top_decile": int(np.percentile(place, 10)),
                            "p_win": float((place == 1).mean()),
                            "p_top5": float((place <= 5).mean())}
                scored.append(rec)

            # Cash ranks on the mean; tournaments rank on the ceiling.
            scored.sort(key=lambda r: -(r["mean"] if kind == "cash"
                                        else r["p_thresh"]))
            report.append((title, kind, field_n, note, scored))
            for rank, rec in enumerate(scored, 1):
                for r in rec["lu"].itertuples(index=False):
                    all_rows.append({
                        "group": kind, "rank": rank, "slot": r.slot,
                        "name": r.name, "team": r.team, "salary": r.salary,
                        "proj": r.proj, "lineup_mean": rec["mean"],
                        "lineup_sd": rec["sd"]})

        if all_rows:
            pd.DataFrame(all_rows).to_csv(f"{base}_lineups.csv", index=False)

        def emit(fh, line):
            print(line)
            fh.write(line + "\n")

        with open(f"{base}.txt", "w", encoding="utf-8") as fh:
            emit(fh, f"Yahoo DFS build {stamp}   contest {args.contest}   "
                     f"cap {cap}")
            for title, kind, field_n, note, scored in report:
                emit(fh, "")
                emit(fh, "=" * 68)
                fld = f"{field_n}-entry field" if field_n else "no field size"
                emit(fh, f"{title}   ({fld}, {note})")
                emit(fh, f"ranked by {'mean projected points' if kind == 'cash' else 'ceiling'}")
                emit(fh, "=" * 68)
                for rank, rec in enumerate(scored, 1):
                    lu = rec["lu"]
                    emit(fh, "")
                    emit(fh, f"--- #{rank} ---  salary "
                             f"{int(lu['salary'].sum())}/{cap}   "
                             f"mean {rec['mean']:.1f} pts   "
                             f"sd {rec['sd']:.1f}   "
                             f"p99 {rec['p99']:.1f}   "
                             f"P(>{args.threshold:.0f}) "
                             f"{100 * rec['p_thresh']:.1f}%")
                    f = rec["finish"]
                    if f:
                        emit(fh, f"      projected finish in {f['n']} entries: "
                                 f"median {f['median']}, "
                                 f"top decile {f['top_decile']}, "
                                 f"P(win) {100 * f['p_win']:.2f}%, "
                                 f"P(top 5) {100 * f['p_top5']:.1f}%")
                    for r in lu.itertuples(index=False):
                        mark = " *" if src.get(r.name) == "projected" else ""
                        nm = (unicodedata.normalize("NFKD", str(r.name))
                              .encode("ascii", "ignore").decode())
                        emit(fh, f"  {r.slot:<3} {nm:<24} {r.team:<4} "
                                 f"{int(r.salary):>3}  {r.proj:>5.1f}{mark}")
            emit(fh, "")
            emit(fh, "  * from a projected lineup (club had not posted)")

        print(f"\nsaved -> {base}.txt")
        print(f"         {base}_lineups.csv")
        print(f"         {base}_pool.csv")
        return


if __name__ == "__main__":
    main()
