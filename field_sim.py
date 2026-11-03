"""Where a lineup finishes in a field, not just what it scores.

A ceiling tells you how often a lineup reaches 170. It does not tell you whether
170 wins, and that depends entirely on what everyone else rosters. If half the
field stacks the same club you did, your good night is their good night and you
split. Raw ceiling optimisation is blind to that.

This estimates the finish distribution directly:

1. **Ownership**, proxied from Yahoo's own published numbers -- the field is
   looking at the site, not running our simulator. Fitted against *real*
   ownership, which Yahoo exposes free through its contest-entry feed
   (`fetch_ownership.py`), so the proxy is calibrated rather than assumed.
   Correlation with truth is about 0.32: right about which players are chalk,
   often wrong about individuals.
2. **A sampled field** of `n_field` opponent lineups drawn from that ownership,
   respecting the same roster rules a real entry must satisfy.
3. **Shared game draws.** Every lineup, ours and theirs, is scored on the *same*
   simulated games. That is what makes duplication cost something: a lineup we
   share players with moves with us, so beating it requires the players we do
   not share.

Finish position is reported rather than prize money, because the payout curve
differs by contest and is a lookup the reader can apply. Field size is an
argument for the same reason.

    python field_sim.py --contest 15744639 --field 360
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def field_ownership(pool, sharpness=2.5):
    """Rough P(rostered by the field), per position.

    Built from Yahoo's projection and salary rather than ours. The field is not
    running our simulator; it is looking at the site, so the site's numbers are
    the better model of what the crowd will do. A softmax within position turns
    value into a share, and `sharpness` controls how concentrated the crowd is
    on the obvious plays -- higher means chalkier.
    """
    d = pool.copy()
    y = pd.to_numeric(d.get("projectedPoints"), errors="coerce")
    # Fall back to our own projection only where Yahoo has none, so the proxy
    # stays a model of the crowd rather than of us.
    y = y.fillna(pd.to_numeric(d.get("proj"), errors="coerce"))
    fppg = pd.to_numeric(d.get("fantasyPointsPerGame"), errors="coerce")

    # Raw projected points, NOT points per dollar. Checked against real
    # ownership from a completed contest (24 entries, Yahoo's own entry feed):
    #
    #   fantasyPointsPerGame        +0.338
    #   projectedPoints             +0.324
    #   projection rank in position +0.303
    #   salary                      +0.294
    #   points per salary dollar    +0.033   <- the original proxy
    #
    # The value-per-dollar version was worthless: Blake Snell was on 50% of
    # entries and it predicted 4.6%; Schwarber 41.7% actual against 0.1%. The
    # field does not hunt value, it takes the best player at each position --
    # the most-owned players ranked 0.72 to 1.00 within position on projection.
    # Refitted on 139 real entries across 5 completed contests (1,003
    # player-rows), which put the pooled correlations at:
    #
    #   rank within position        +0.318
    #   fantasyPointsPerGame        +0.302
    #   0.5*proj + 0.5*fppg         +0.302
    #   projectedPoints             +0.272
    #   salary                      +0.264
    #   points per salary dollar    +0.019
    #
    # Rank wins narrowly and is the more robust choice: it normalises across
    # contests whose scoring levels differ, so the same proxy works on a
    # two-game slate and an eleven-game one.
    #
    # A single contest had suggested +0.368; pooling brought that to +0.32, so
    # treat ~0.3 as the honest figure. This model gets the broad shape of chalk
    # right and individual players often wrong -- it is a weak estimate, and
    # finish positions built on it inherit that.
    base = (0.5 * y.fillna(0.0) + 0.5 * fppg.fillna(y).fillna(0.0))
    d["_base"] = base
    d["own_score"] = d.groupby("primaryPosition")["_base"].rank(pct=True)
    d = d.drop(columns=["_base"])

    out = np.zeros(len(d))
    for pos, g in d.groupby("primaryPosition"):
        v = g["own_score"].to_numpy()
        if not len(v) or v.std() == 0:
            share = np.full(len(v), 1.0 / max(len(v), 1))
        else:
            z = sharpness * (v - v.mean()) / (v.std() + 1e-9)
            e = np.exp(z - z.max())
            share = e / e.sum()
        out[d.index.get_indexer(g.index)] = share
    d["own"] = out
    return d


def sample_field(pool, cap, n_field, roster, rng, max_per_team=6,
                 min_teams=3, max_tries=25):
    """Draw plausible opponent lineups from the ownership model.

    Salary-aware construction rather than rejection sampling. Slots are filled
    in random order, and each pick is drawn only from players still affordable
    once a minimum is reserved for the slots left to fill. The earlier version
    drew ten players at random and then tested the cap, which succeeded about
    12% of the time and cost two seconds per requested entry -- a 360-entry
    field alone ran longer than everything else in the build.

    Team rules are enforced during construction too, so a draw cannot paint
    itself into an illegal corner and be thrown away at the end.
    """
    d = field_ownership(pool)
    d = d[pd.to_numeric(d["salary"], errors="coerce").notna()]
    by_pos = {p: g.reset_index(drop=True) for p, g in d.groupby("primaryPosition")}
    # Cheapest available at each position, to reserve a floor for unfilled slots.
    floor = {p: float(g["salary"].min()) for p, g in by_pos.items()}

    slots = [p for p, n in roster.items() for _ in range(n)]
    out = []
    for _ in range(n_field):
        for _try in range(max_tries):
            order = list(slots)
            rng.shuffle(order)
            picked, used, spend = [], set(), 0.0
            teams = {}
            ok = True
            for k, pos in enumerate(order):
                g = by_pos.get(pos)
                if g is None or not len(g):
                    ok = False
                    break
                # Reserve the cheapest option for every slot still to come.
                reserve = sum(floor.get(q, 0.0) for q in order[k + 1:])
                budget = cap - spend - reserve
                sal = pd.to_numeric(g["salary"], errors="coerce")
                avail = (sal <= budget) & (~g.index.isin(used))
                # Respect the club cap while drawing, not afterwards.
                full = {t for t, c in teams.items() if c >= max_per_team}
                if full:
                    avail &= ~g["team"].isin(full)
                cand = g[avail]
                if not len(cand):
                    ok = False
                    break
                w = cand["own"].to_numpy(dtype=float)
                w = w / w.sum() if w.sum() > 0 else None
                j = rng.choice(len(cand), p=w)
                row = cand.iloc[j]
                picked.append(row)
                used.add(cand.index[j])
                spend += float(row["salary"])
                teams[row["team"]] = teams.get(row["team"], 0) + 1
            if not ok or len(picked) != len(slots):
                continue
            if len(teams) < min_teams:
                continue
            ids = [int(r["mlbam"]) for r in picked if pd.notna(r.get("mlbam"))]
            if len(ids) == len(slots):
                out.append(ids)
                break
    return out


def score_matrix(samples, player_ids, n_draws=4000, seed=7):
    """Points for every player on every draw, correlation preserved.

    One shared simulation index per game, drawn independently across games, so
    teammates move together and unrelated games do not. Returns (matrix, index)
    where the matrix is n_draws x len(index).
    """
    rng = np.random.default_rng(seed)
    by_game = {}
    for pid in player_ids:
        got = samples.get(int(pid))
        if got is None:
            continue
        by_game.setdefault(got[0], []).append(int(pid))
    if not by_game:
        return None, {}

    index = {}
    cols = []
    for gk, pids in by_game.items():
        n = min(len(samples[p][1]) for p in pids)
        idx = rng.integers(0, n, size=n_draws)
        for p in pids:
            index[p] = len(cols)
            cols.append(np.asarray(samples[p][1], dtype=float)[idx])
    return np.column_stack(cols), index


def finish_distribution(my_ids, field, samples, n_draws=4000, seed=7):
    """Where our lineup lands, draw by draw, against the sampled field."""
    everyone = set(int(x) for x in my_ids)
    for lu in field:
        everyone.update(int(x) for x in lu)
    mat, index = score_matrix(samples, everyone, n_draws, seed)
    if mat is None:
        return None

    def total(ids):
        cols = [index[int(i)] for i in ids if int(i) in index]
        if not cols:
            return None
        return mat[:, cols].sum(axis=1)

    mine = total(my_ids)
    if mine is None:
        return None
    beats = np.zeros(n_draws, dtype=np.int32)
    scored = 0
    for lu in field:
        t = total(lu)
        if t is None:
            continue
        scored += 1
        beats += (t > mine)
    # Finish position: one plus the number of entries that outscored us.
    return beats + 1, scored
