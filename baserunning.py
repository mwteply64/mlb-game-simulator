"""Empirical base-out state transitions.

Given an outcome, the base state, and the number of outs, what happens? Rather
than encoding advancement rules by hand -- which is where homemade simulators
quietly lose runs -- this counts the joint distribution of
(runs scored, outs recorded, resulting base state) straight from a million
observed plate appearances.

Counting the joint distribution empirically gets the rule subtleties right for
free. Runs that do not count because the third out was a force play are simply
absent from the data. Sacrifice flies, fielder's choices, double plays, and
runners taking an extra base on a throw are all already in the proportions
they actually occur in.

The tables are flattened into a form the simulator can sample from with a
single vectorized `searchsorted`, which is what makes tens of millions of
simulated plate appearances practical in NumPy.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)
N_BASE = 8   # base occupancy bitmask: 1B=1, 2B=2, 3B=4
N_OUTS = 3

# Transitions are conditioned on two speed terciles, because the two
# baserunning channels are driven by different players. Advancement -- going
# first to third on a single, scoring from first on a double -- depends on the
# runners already on base. Double plays depend on the batter running to first.
# Measured over 2021-2026 both effects are large: first-to-third runs 18.9% for
# the slowest tercile against 30.7% for the fastest, scoring from first on a
# double 21.0% against 46.0%, and the double-play rate falls from 27.4% to
# 19.1% across batter speed terciles.
N_SPEED = 3
N_CELLS = N_OUT * N_BASE * N_OUTS * N_SPEED * N_SPEED

# Runs are capped when building the table; more than four on a single plate
# appearance is a grand slam and nothing else.
MAX_RUNS = 4


def cell_index(outcome_idx, base_code, outs, runner_bucket=1, batter_bucket=1):
    return ((((outcome_idx * N_BASE + base_code) * N_OUTS + outs)
             * N_SPEED + runner_bucket) * N_SPEED + batter_bucket)


class TransitionTable:
    """Flattened empirical transitions, sampled by one global searchsorted.

    Each cell's cumulative probabilities are stored offset by the cell index,
    so the whole array is globally monotonic: cell c occupies values in
    [c, c+1). Sampling a uniform u for a sim in cell c means searching for
    c + u across the entire array at once, which vectorizes over every
    simulation regardless of which cell each one is in.
    """

    def __init__(self, cum, runs, outs_added, base_after, counts, starts,
                 fallback_mask, speed_edges=None):
        self.cum = cum
        self.runs = runs
        self.outs_added = outs_added
        self.base_after = base_after
        self.counts = counts
        self.starts = starts
        self.fallback_mask = fallback_mask
        # Tercile boundaries used to bucket runner and batter speed, so the
        # simulator buckets with exactly the thresholds the table was built on.
        self.speed_edges = speed_edges or (np.array([]), np.array([]))

    def sample(self, cells, u):
        """Vectorized draw. `cells` and `u` are parallel arrays."""
        idx = np.searchsorted(self.cum, cells + u, side="left")
        np.clip(idx, 0, len(self.cum) - 1, out=idx)
        return self.runs[idx], self.outs_added[idx], self.base_after[idx]

    def save(self, path):
        np.savez_compressed(
            path, cum=self.cum, runs=self.runs, outs_added=self.outs_added,
            base_after=self.base_after, counts=self.counts, starts=self.starts,
            fallback_mask=self.fallback_mask,
            runner_edges=self.speed_edges[0], batter_edges=self.speed_edges[1])

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["cum"], z["runs"], z["outs_added"], z["base_after"],
                   z["counts"], z["starts"], z["fallback_mask"],
                   speed_edges=(z["runner_edges"], z["batter_edges"]))


def _deterministic_rows(outcome_idx, base_code, outs):
    """Fallback transitions for cells with no observations.

    Only reached for combinations that essentially never occur (a triple with
    the bases loaded and two outs, say). Encodes the plain baseball rule so the
    simulator never has an undefined state.
    """
    o = OUTCOMES[outcome_idx]
    on1 = base_code & 1
    on2 = (base_code >> 1) & 1
    on3 = (base_code >> 2) & 1
    n_on = on1 + on2 + on3

    if o == "HR":
        return [(min(n_on + 1, MAX_RUNS), 0, 0)]
    if o in ("BB", "HBP"):
        # Forced advances only.
        if on1 and on2 and on3:
            return [(1, 0, 7)]
        if on1 and on2:
            return [(0, 0, 7)]
        if on1:
            return [(0, 0, base_code | 2 | 1)]
        return [(0, 0, base_code | 1)]
    if o == "K":
        return [(0, 1, base_code)]
    if o == "OUT":
        return [(0, 1, base_code)]
    if o == "1B":
        return [(on3, 0, ((base_code << 1) & 6) | 1)]
    if o == "2B":
        return [(on3 + on2, 0, ((on1 * 4) | 2))]
    if o == "3B":
        return [(n_on, 0, 4)]
    return [(0, 1, base_code)]


def speed_buckets(d, speed_ratings, n=N_SPEED):
    """Runner and batter speed terciles for each plate appearance.

    The runner bucket is the mean projected sprint speed across occupied bases.
    With the bases empty there is no runner to condition on, so it falls to the
    middle bucket and the batter bucket carries whatever speed signal exists.
    """
    if speed_ratings is None:
        z = np.full(len(d), n // 2, dtype=np.int8)
        return z, z.copy()

    def spd(col):
        ids = d[col].to_numpy()
        out = np.full(len(d), np.nan)
        for i, p in enumerate(ids):
            if p == p:
                out[i] = speed_ratings.get(int(p))
        return out

    runner = np.nanmean(np.column_stack(
        [spd("r1_before"), spd("r2_before"), spd("r3_before")]), axis=1)
    batter = np.array([speed_ratings.get(int(p)) for p in d["batter"]])

    valid = runner[~np.isnan(runner)]
    r_edges = (np.quantile(valid, [1 / n * i for i in range(1, n)])
               if len(valid) else np.array([0.0]))
    b_edges = np.quantile(batter, [1 / n * i for i in range(1, n)])

    r_bucket = np.where(np.isnan(runner), n // 2,
                        np.searchsorted(r_edges, runner)).astype(np.int8)
    b_bucket = np.searchsorted(b_edges, batter).astype(np.int8)
    return np.clip(r_bucket, 0, n - 1), np.clip(b_bucket, 0, n - 1), r_edges, b_edges


def build(pa, seasons=None, min_count=25, speed_ratings=None):
    """Count transitions from the plate appearance table.

    When `speed_ratings` is supplied, transitions are conditioned on runner and
    batter speed terciles, with hierarchical fallback: a thin
    (cell, runner, batter) combination backs off to (cell, runner), then to the
    speed-agnostic cell, then to rule-based defaults.
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]

    d["outs_before"] = d["outs_before"].astype(int)
    d["outs_pa"] = d["outs_pa"].fillna(0).astype(int)
    d["runs_pa"] = d["runs_pa"].fillna(0).clip(0, MAX_RUNS).astype(int)
    d["base_code_before"] = d["base_code_before"].astype(int)

    # Resulting base state: -1 marks a half-inning that ended, which the
    # simulator represents as bases empty.
    base_after = d["base_code_after"].fillna(-1).astype(int)
    d["base_after"] = np.where(base_after < 0, 0, base_after)

    # Discard rows whose out accounting is internally inconsistent: a
    # half-inning that ended without reaching three outs is a walk-off, where
    # the game stopped rather than the inning.
    total_outs = d["outs_before"] + d["outs_pa"]
    consistent = (total_outs <= 3) & (~d["ends_half"] | (total_outs == 3))
    dropped = (~consistent).sum()
    d = d[consistent]
    print(f"transitions: {len(d):,} plate appearances "
          f"({dropped:,} dropped as walk-off / inconsistent)")

    d["outcome_idx"] = d["outcome"].map({o: i for i, o in enumerate(OUTCOMES)})

    if speed_ratings is not None:
        r_bucket, b_bucket, r_edges, b_edges = speed_buckets(d, speed_ratings)
    else:
        r_bucket = np.full(len(d), N_SPEED // 2, dtype=np.int8)
        b_bucket = r_bucket.copy()
        r_edges = b_edges = np.array([])
    d["r_bucket"], d["b_bucket"] = r_bucket, b_bucket

    d["cell"] = cell_index(d["outcome_idx"].to_numpy(),
                           d["base_code_before"].to_numpy(),
                           d["outs_before"].to_numpy(),
                           r_bucket, b_bucket)
    # Backoff keys: same base-out cell, ignoring one or both speed dimensions.
    d["cell_r"] = cell_index(d["outcome_idx"].to_numpy(),
                             d["base_code_before"].to_numpy(),
                             d["outs_before"].to_numpy(),
                             r_bucket, N_SPEED // 2)
    d["cell_0"] = cell_index(d["outcome_idx"].to_numpy(),
                             d["base_code_before"].to_numpy(),
                             d["outs_before"].to_numpy(),
                             N_SPEED // 2, N_SPEED // 2)

    key = ["runs_pa", "outs_pa", "base_after"]
    grp = d.groupby(["cell"] + key).size().rename("n").reset_index()
    grp_r = (d.groupby(["cell_r"] + key).size().rename("n").reset_index()
             .rename(columns={"cell_r": "cell"}))
    grp_0 = (d.groupby(["cell_0"] + key).size().rename("n").reset_index()
             .rename(columns={"cell_0": "cell"}))

    cum_l, runs_l, oa_l, ba_l = [], [], [], []
    counts = np.zeros(N_CELLS, dtype=np.int32)
    starts = np.zeros(N_CELLS, dtype=np.int32)
    fallback = np.zeros(N_CELLS, dtype=bool)
    backoff_counts = {"exact": 0, "runner_only": 0, "agnostic": 0, "rule": 0}

    by_cell = {c: g for c, g in grp.groupby("cell")}
    by_cell_r = {c: g for c, g in grp_r.groupby("cell")}
    by_cell_0 = {c: g for c, g in grp_0.groupby("cell")}

    cursor = 0
    for c in range(N_CELLS):
        rem, bb = divmod(c, N_SPEED)
        rem, rb = divmod(rem, N_SPEED)
        rem, ob = divmod(rem, N_OUTS)
        oi, bc = divmod(rem, N_BASE)

        g = by_cell.get(c)
        if g is not None and g["n"].sum() >= min_count:
            backoff_counts["exact"] += 1
        else:
            g = by_cell_r.get(cell_index(oi, bc, ob, rb, N_SPEED // 2))
            if g is not None and g["n"].sum() >= min_count:
                backoff_counts["runner_only"] += 1
            else:
                g = by_cell_0.get(cell_index(oi, bc, ob))
                if g is not None and g["n"].sum() >= min_count:
                    backoff_counts["agnostic"] += 1
                else:
                    g = None

        if g is None:
            rows = _deterministic_rows(oi, bc, ob)
            probs = np.ones(len(rows)) / len(rows)
            fallback[c] = True
            backoff_counts["rule"] += 1
            r = np.array([x[0] for x in rows], dtype=np.int8)
            a = np.array([x[1] for x in rows], dtype=np.int8)
            b = np.array([x[2] for x in rows], dtype=np.int8)
        else:
            n = g["n"].to_numpy(dtype=float)
            probs = n / n.sum()
            r = g["runs_pa"].to_numpy(dtype=np.int8)
            a = g["outs_pa"].to_numpy(dtype=np.int8)
            b = g["base_after"].to_numpy(dtype=np.int8)

        # Offset the cumulative probabilities by the cell index so the global
        # array stays sorted and one searchsorted serves every cell.
        cp = np.cumsum(probs)
        cp[-1] = 1.0
        cum_l.append(cp + c)
        runs_l.append(r)
        oa_l.append(a)
        ba_l.append(b)

        starts[c] = cursor
        counts[c] = len(probs)
        cursor += len(probs)

    table = TransitionTable(
        np.concatenate(cum_l), np.concatenate(runs_l).astype(np.int8),
        np.concatenate(oa_l).astype(np.int8),
        np.concatenate(ba_l).astype(np.int8), counts, starts, fallback)
    table.speed_edges = (r_edges, b_edges)

    print(f"  {N_CELLS} cells, {len(table.cum):,} distinct transitions")
    print(f"  backoff: {backoff_counts['exact']} exact, "
          f"{backoff_counts['runner_only']} runner-only, "
          f"{backoff_counts['agnostic']} speed-agnostic, "
          f"{backoff_counts['rule']} rule-based")
    return table


def run_expectancy(pa, seasons=None):
    """The 24-state run expectancy matrix: mean runs scored from each
    (base state, outs) to the end of the half-inning.

    Only half-innings that recorded three outs are used. A half-inning ended by
    a walk-off stopped early by rule, so counting it would understate the runs
    the state was worth.

    Returns an array indexed [base_code, outs].
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]

    d["runs_pa"] = d["runs_pa"].fillna(0)
    g = d.groupby("half_key")
    complete = g["outs_pa"].transform("sum") == 3
    d = d[complete]

    g = d.groupby("half_key")
    total = g["runs_pa"].transform("sum")
    runs_before = g["runs_pa"].cumsum() - d["runs_pa"]
    d["runs_rest"] = total - runs_before

    re = np.zeros((N_BASE, N_OUTS))
    m = d.groupby([d["base_code_before"].astype(int),
                   d["outs_before"].astype(int)])["runs_rest"].mean()
    for (bc, ob), v in m.items():
        if 0 <= bc < N_BASE and 0 <= ob < N_OUTS:
            re[bc, ob] = v
    return re


def linear_weights(pa, re, seasons=None):
    """Empirical run value of each outcome: runs scored on the play plus the
    change in run expectancy it produced.

    Derived from this project's own data rather than imported constants, so the
    weights match the run environment the simulator is reproducing.
    """
    d = pa[pa["is_pa"] & pa["outcome"].isin(OUTCOMES)].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]

    d["runs_pa"] = d["runs_pa"].fillna(0)
    bc_b = d["base_code_before"].astype(int).to_numpy()
    ob_b = d["outs_before"].astype(int).to_numpy()
    re_before = re[bc_b, ob_b]

    ends = d["ends_half"].to_numpy()
    bc_a = np.clip(d["base_code_after"].fillna(0).astype(int).to_numpy(), 0, N_BASE - 1)
    ob_a = np.clip(d["outs_after"].fillna(3).astype(int).to_numpy(), 0, N_OUTS - 1)
    re_after = np.where(ends, 0.0, re[bc_a, ob_a])

    d["run_value"] = d["runs_pa"].to_numpy() + re_after - re_before
    lw = d.groupby("outcome")["run_value"].mean()
    return np.array([lw.get(o, 0.0) for o in OUTCOMES])


def _cell_probs(table, c):
    s, n = table.starts[c], table.counts[c]
    return np.diff(np.concatenate([[float(c)], table.cum[s:s + n]])), s, n


def expected_runs(table, oi, bc, ob, rb=1, bb=1):
    c = cell_index(oi, bc, ob, rb, bb)
    p, s, n = _cell_probs(table, c)
    return float(np.dot(p, table.runs[s:s + n]))


def summarize(table):
    """Expected runs per outcome, and the size of the speed effect."""
    print("\nexpected runs on the play, by outcome and base state "
          "(0 outs, average speed):")
    for oi, o in enumerate(OUTCOMES):
        parts = [f"{bc}:{expected_runs(table, oi, bc, 0):.2f}"
                 for bc in range(N_BASE)]
        print(f"  {o:<4} " + "  ".join(parts))

    print("\nspeed effect -- expected runs on the play, slow vs fast:")
    checks = [("1B", 1, 0, "runner on 1st, single"),
              ("2B", 1, 0, "runner on 1st, double"),
              ("1B", 2, 0, "runner on 2nd, single"),
              ("1B", 3, 0, "runners on 1st+2nd, single")]
    for o, bc, ob, label in checks:
        oi = OUTCOMES.index(o)
        slow = expected_runs(table, oi, bc, ob, rb=0, bb=1)
        fast = expected_runs(table, oi, bc, ob, rb=2, bb=1)
        print(f"  {label:<28} slow {slow:.3f}  fast {fast:.3f}  "
              f"({fast - slow:+.3f})")

    oi = OUTCOMES.index("OUT")
    print("\n  double-play rate by batter speed (runner on 1st, 0 outs):")
    for bb, lbl in [(0, "slow"), (2, "fast")]:
        c = cell_index(oi, 1, 0, 1, bb)
        p, s, n = _cell_probs(table, c)
        dp = float(p[table.outs_added[s:s + n] >= 2].sum())
        print(f"    {lbl}: {dp * 100:.2f}%")


def main():
    import outcome_model as om
    import player_ratings as pr

    pa = om.load_pa()
    _, speed = pr.load_ratings()
    speed_ratings = pr.SpeedRatings(speed, max(config.SEASONS) + 1)

    table = build(pa, seasons=config.TACTICAL_SEASONS,
                  speed_ratings=speed_ratings)
    path = os.path.join(config.PROCESSED_DIR, "transitions.npz")
    table.save(path)
    print(f"\nwrote {path}")
    summarize(table)


if __name__ == "__main__":
    main()
