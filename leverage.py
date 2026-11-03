"""Win probability by game state, and leverage anywhere in a game.

Leverage is how much the next plate appearance can swing the game. It is what
tells a manager where a decision is worth spending a reliever or a bench bat,
and it is the denominator every decision should be read against: a two-point
edge in a high-leverage spot is worth several times the same edge in a blowout.

Two pieces:

* **A win probability table**, counted empirically over every plate appearance
  in the dataset, indexed by inning, half, outs, base state, and score margin.
* **A leverage function** that weights the win-probability swing of each
  possible outcome by how likely that outcome is *in this matchup*.

That second point is the one that matters and the one a published leverage
table cannot do. Standard Leverage Index is a property of the base-out-inning
state alone, so it says the same thing whether the hitter is the league's best
bat or its worst. Feeding a matchup-specific outcome distribution makes
leverage depend on who is actually involved -- a slugger facing a tiring
right-hander genuinely creates more leverage than a light-hitting catcher in the
same base-out state, because the distribution of what might happen is wider.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import baserunning
import config

OUTCOMES = config.OUTCOMES
N_OUT = len(OUTCOMES)

MAX_INNING = 10          # ninth and beyond collapse into one bucket
MAX_MARGIN = 8           # margins beyond this are effectively decided
N_MARGIN = 2 * MAX_MARGIN + 1
N_BASE = 8
N_OUTS = 3

WP_PATH = os.path.join(config.PROCESSED_DIR, "win_probability.npz")
MIN_COUNT = 50           # below this, back off to a coarser state


def state_index(inning, is_top, outs, base_code, margin):
    """Flat index into the win probability table.

    `margin` is from the batting team's perspective, clipped: beyond eight runs
    the game is decided and the extra detail is noise.
    """
    inn = np.clip(inning, 1, MAX_INNING) - 1
    m = np.clip(margin, -MAX_MARGIN, MAX_MARGIN) + MAX_MARGIN
    half = np.where(is_top, 0, 1)
    return (((inn * 2 + half) * N_OUTS + outs) * N_BASE + base_code) * N_MARGIN + m


N_STATES = MAX_INNING * 2 * N_OUTS * N_BASE * N_MARGIN


def build_win_probability(pa, seasons=None):
    """Empirical probability the BATTING team wins, from each state.

    Backs off to progressively coarser states where the sample is thin: first
    ignoring the base state, then ignoring outs. Rare states -- a nine-run lead
    with the bases loaded in the second -- would otherwise be estimated from a
    handful of games and produce nonsense leverage.
    """
    d = pa[pa["is_pa"]].copy()
    if seasons:
        d = d[d["season"].isin(seasons)]

    final = (d.groupby(["game_pk", "batting_team"])["runs_pa"].sum()
             .rename("final_runs").reset_index())
    d = d.merge(final, on=["game_pk", "batting_team"], how="left")
    opp = final.rename(columns={"batting_team": "fielding_team",
                                "final_runs": "opp_runs"})
    d = d.merge(opp, on=["game_pk", "fielding_team"], how="left")

    d = d.dropna(subset=["final_runs", "opp_runs"])
    d["bat_wins"] = (d["final_runs"] > d["opp_runs"]).astype(float)
    d["margin"] = (d["bat_score_before"] - d["fld_score_before"]).astype(int)

    idx = state_index(d["inning"].to_numpy(), d["is_top"].to_numpy(),
                      d["outs_before"].astype(int).to_numpy(),
                      d["base_code_before"].astype(int).to_numpy(),
                      d["margin"].to_numpy())

    wins = np.bincount(idx, weights=d["bat_wins"].to_numpy(), minlength=N_STATES)
    n = np.bincount(idx, minlength=N_STATES).astype(float)

    # Coarser backoffs, computed the same way.
    idx_nb = state_index(d["inning"].to_numpy(), d["is_top"].to_numpy(),
                         d["outs_before"].astype(int).to_numpy(),
                         np.zeros(len(d), dtype=int), d["margin"].to_numpy())
    w_nb = np.bincount(idx_nb, weights=d["bat_wins"].to_numpy(), minlength=N_STATES)
    n_nb = np.bincount(idx_nb, minlength=N_STATES).astype(float)

    idx_no = state_index(d["inning"].to_numpy(), d["is_top"].to_numpy(),
                         np.zeros(len(d), dtype=int),
                         np.zeros(len(d), dtype=int), d["margin"].to_numpy())
    w_no = np.bincount(idx_no, weights=d["bat_wins"].to_numpy(), minlength=N_STATES)
    n_no = np.bincount(idx_no, minlength=N_STATES).astype(float)

    wp = np.full(N_STATES, np.nan)
    ok = n >= MIN_COUNT
    wp[ok] = wins[ok] / n[ok]

    # Backoff 1: same inning/half/outs/margin, base state ignored.
    need = ~ok
    for lvl_w, lvl_n, lvl_idx in ((w_nb, n_nb, idx_nb), (w_no, n_no, idx_no)):
        coarse = np.full(N_STATES, np.nan)
        good = lvl_n >= MIN_COUNT
        coarse[good] = lvl_w[good] / lvl_n[good]
        # Map every fine state onto its coarse counterpart.
        inn = np.arange(N_STATES) // (2 * N_OUTS * N_BASE * N_MARGIN)
        rem = np.arange(N_STATES) % (2 * N_OUTS * N_BASE * N_MARGIN)
        half = rem // (N_OUTS * N_BASE * N_MARGIN)
        rem2 = rem % (N_OUTS * N_BASE * N_MARGIN)
        outs = rem2 // (N_BASE * N_MARGIN)
        rem3 = rem2 % (N_BASE * N_MARGIN)
        marg = rem3 % N_MARGIN
        if lvl_idx is idx_nb:
            tgt = (((inn * 2 + half) * N_OUTS + outs) * N_BASE + 0) * N_MARGIN + marg
        else:
            tgt = (((inn * 2 + half) * N_OUTS + 0) * N_BASE + 0) * N_MARGIN + marg
        fill = need & np.isfinite(coarse[tgt])
        wp[fill] = coarse[tgt][fill]
        need = ~np.isfinite(wp)

    # Anything still missing: the margin alone decides it.
    missing = ~np.isfinite(wp)
    margins = (np.arange(N_STATES) % N_MARGIN) - MAX_MARGIN
    wp[missing] = np.clip(0.5 + 0.06 * margins[missing], 0.02, 0.98)

    np.savez_compressed(WP_PATH, wp=wp, n=n)
    print(f"win probability table: {int(ok.sum()):,} of {N_STATES:,} states "
          f"estimated directly, rest backed off -> {WP_PATH}")
    return wp


def load_win_probability():
    if not os.path.exists(WP_PATH):
        raise SystemExit(f"no table at {WP_PATH}; run build_win_probability first")
    return np.load(WP_PATH)["wp"]


def wp_after(wp, inning, is_top, outs, base_code, margin, runs, outs_added,
             base_after):
    """Win probability for the batting team after an outcome resolves."""
    new_outs = outs + outs_added
    new_margin = margin + runs
    ends = new_outs >= 3
    # When the half-inning ends the other side bats, so the perspective flips.
    nxt_inn = np.where(ends & ~is_top, inning + 1, inning)
    nxt_top = np.where(ends, ~is_top, is_top)
    nxt_outs = np.where(ends, 0, new_outs)
    nxt_base = np.where(ends, 0, base_after)
    nxt_margin = np.where(ends, -new_margin, new_margin)

    idx = state_index(nxt_inn, nxt_top, nxt_outs.astype(int),
                      nxt_base.astype(int), nxt_margin)
    p = wp[idx]
    # Flipped perspective means the batting team's win probability is 1 - p.
    return np.where(ends, 1.0 - p, p)


def leverage(wp, table, inning, is_top, outs, base_code, margin,
             outcome_probs=None, runner_bucket=1, batter_bucket=1):
    """Leverage: expected absolute swing in win probability on this plate
    appearance, in units where the league average state is 1.0.

    Pass `outcome_probs` from the matchup to make this depend on who is batting
    and who is pitching. Left as None it uses league rates and reproduces a
    conventional, matchup-blind leverage index.
    """
    if outcome_probs is None:
        outcome_probs = LEAGUE_OUTCOME_RATES

    here = wp[state_index(inning, is_top, outs, base_code, margin)]
    swing = 0.0
    for oi in range(N_OUT):
        c = baserunning.cell_index(oi, base_code, outs, runner_bucket,
                                   batter_bucket)
        s, cnt = table.starts[c], table.counts[c]
        probs = np.diff(np.concatenate([[float(c)], table.cum[s:s + cnt]]))
        after = wp_after(wp, np.array([inning]), np.array([is_top]),
                         np.array([outs]), np.array([base_code]),
                         np.array([margin]),
                         table.runs[s:s + cnt].astype(int),
                         table.outs_added[s:s + cnt].astype(int),
                         table.base_after[s:s + cnt].astype(int))
        swing += outcome_probs[oi] * float(np.dot(probs, np.abs(after - here)))
    return swing


LEAGUE_OUTCOME_RATES = np.array(
    [0.2258, 0.0847, 0.0112, 0.4596, 0.1412, 0.0430, 0.0037, 0.0308])

LI_TABLE_PATH = os.path.join(config.PROCESSED_DIR, "leverage_table.npz")


def build_leverage_table(wp, table, path=LI_TABLE_PATH):
    """Leverage for every state, precomputed once and cached.

    `leverage()` is scalar and walks the transition table per call, which is far
    too slow to evaluate per plate appearance -- let alone inside the simulation
    loop, where it would fire for every batter in every sim. There are only
    8,160 states, so computing all of them once and indexing is the obvious
    trade: identical numbers, no per-call cost.
    """
    li = np.ones(N_STATES, dtype=float)
    for inning in range(1, MAX_INNING + 1):
        for half in (True, False):
            for outs in range(N_OUTS):
                for base in range(N_BASE):
                    for m in range(-MAX_MARGIN, MAX_MARGIN + 1):
                        idx = int(state_index(inning, half, outs, base, m))
                        li[idx] = leverage(wp, table, inning, half, outs,
                                           base, m)
    mean = li[li > 0].mean()
    if mean > 0:
        li = li / mean          # league-average state == 1.0 by construction
    np.savez_compressed(path, li=li)
    return li


def load_leverage_table(path=LI_TABLE_PATH):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)["li"]
    except Exception:
        return None


def main():
    import outcome_model as om
    import player_ratings as pr

    pa = om.load_pa()
    wp = build_win_probability(pa)

    _, speed = pr.load_ratings()
    table = baserunning.build(pa, seasons=config.TACTICAL_SEASONS,
                              speed_ratings=pr.SpeedRatings(speed, 2026))

    print("\nwin probability for the batting team, tie game, bases empty:")
    for inn in (1, 5, 7, 9):
        row = []
        for outs in range(3):
            p = wp[state_index(inn, True, outs, 0, 0)]
            row.append(f"{outs} out {p * 100:5.1f}%")
        print(f"  inning {inn}, top: " + "   ".join(row))

    print("\nleverage by situation (league-average matchup):")
    situations = [
        ("1st inning, bases empty, tie", 1, True, 0, 0, 0),
        ("5th, runner on 1st, tie", 5, True, 1, 1, 0),
        ("7th, runner on 3rd, 2 out, down 1", 7, True, 2, 4, -1),
        ("9th, bases loaded, 2 out, down 1", 9, False, 2, 7, -1),
        ("9th, bases empty, up 7", 9, False, 0, 0, 7),
    ]
    raw = {}
    for label, inn, top, outs, base, marg in situations:
        raw[label] = leverage(wp, table, inn, top, outs, base, marg)

    # Normalise so an average plate appearance is 1.0.
    ref = leverage(wp, table, 5, True, 1, 0, 0)
    for label, v in raw.items():
        print(f"  {label:<38} {v / ref:5.2f}")
    print(f"\n  (normalised against a 5th-inning, one-out, bases-empty tie)")


if __name__ == "__main__":
    main()
