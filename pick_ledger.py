"""A running record of what the model picked and how it did.

Every live run writes its picks here before the games are played, and every
later run grades whatever has since settled. That ordering is the whole point:
a pick is recorded with a timestamp ahead of first pitch, so the record cannot
be quietly improved after the fact. It is the same discipline the odds recorder
uses, applied to the model's own output.

Reported three ways -- yesterday, the last seven days, and everything -- plus
the same figures restricted to picks the model made at 60% confidence or better.
That split is the useful one: if the model knows anything, its confident picks
should beat its marginal ones, and if they do not, the probabilities are not
carrying information even when the overall rate looks fine.

**Read the win rates against the base rate, not against 52.4%.** A market where
the model takes the same side almost every time will post a high win rate purely
from the base rate of that side -- total bases won 60.0% while blind unders won
59.7%. The grader records which side was taken so that comparison stays
possible, but the honest benchmark is per market and is reported by
`check_prop_lean.py`.

    python pick_ledger.py --report
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

PATH = os.path.join(config.PROCESSED_DIR, "pick_ledger.parquet")
CONFIDENT = 0.60


def record(picks):
    """Append picks. `picks` needs game_pk, game_date, player, market, line,
    side, p_side (the model's probability for the side it took)."""
    if not len(picks):
        return
    df = pd.DataFrame(picks)
    df["recorded_utc"] = pd.Timestamp.utcnow()
    if "result" not in df.columns:
        df["result"] = np.nan
    if os.path.exists(PATH):
        old = pd.read_parquet(PATH)
        df = pd.concat([old, df], ignore_index=True)
        # One row per pick; a rerun on the same day refreshes rather than
        # duplicating, keeping the newest projection for that market.
        df = df.drop_duplicates(subset=["game_pk", "player", "market"],
                                keep="last")
    df.to_parquet(PATH, index=False)
    return len(df)


def grade(pa=None):
    """Fill in results for any pick whose game has since been played."""
    if not os.path.exists(PATH):
        return None
    led = pd.read_parquet(PATH)
    todo = led[led["result"].isna()]
    if not len(todo):
        return led

    if pa is None:
        import outcome_model as om
        pa = om.load_pa()
    d = pa[pa["is_pa"]]

    w = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
    bat = d.assign(tb=d["outcome"].map(w).fillna(0),
                   hit=d["outcome"].isin(["1B", "2B", "3B", "HR"]).astype(int))
    bat = bat.groupby(["game_pk", "batter"]).agg(
        tb=("tb", "sum"), hits=("hit", "sum")).reset_index()
    st = d[d["is_starter"]]
    pit = st.groupby(["game_pk", "pitcher"]).agg(
        k=("outcome", lambda s: (s == "K").sum()),
        outs=("outs_pa", "sum")).reset_index()

    actual = {}
    for r in bat.itertuples(index=False):
        actual[(r.game_pk, r.batter, "total bases")] = r.tb
        actual[(r.game_pk, r.batter, "hits")] = r.hits
    for r in pit.itertuples(index=False):
        actual[(r.game_pk, r.pitcher, "strikeouts")] = r.k
        actual[(r.game_pk, r.pitcher, "outs recorded")] = r.outs

    res = []
    for r in led.itertuples(index=False):
        if not np.isnan(r.result):
            res.append(r.result)
            continue
        a = actual.get((int(r.game_pk), int(r.player), r.market))
        if a is None:
            res.append(np.nan)
        elif a == r.line:
            res.append(0.5)                      # push
        else:
            hit = (a > r.line) if r.side == "over" else (a < r.line)
            res.append(1.0 if hit else 0.0)
    led["result"] = res
    led.to_parquet(PATH, index=False)
    return led


def blind_rate(led, pa=None):
    """Base rate of the sides the model actually took, per market.

    The benchmark that matters. Comparing a win rate to 52.4% only works when
    the line is near a coin flip; when the model takes one side most of the
    time, the honest question is whether it beats taking that side blindly.
    Computed from history rather than pinned to a constant, because it moves
    with the model's lean -- 54.2% for strikeouts at a 69% under lean, and a
    different number if that lean changes.
    """
    if pa is None:
        import outcome_model as om
        pa = om.load_pa()
    d = pa[pa["is_pa"]]
    st = d[d["is_starter"]]
    k = st.groupby(["game_pk", "pitcher"]).apply(
        lambda x: (x["outcome"] == "K").sum()).rename("v").reset_index()
    outs = (st.groupby(["game_pk", "pitcher"])["outs_pa"].sum()
            .rename("v").reset_index())
    w = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
    b = d.assign(tb=d["outcome"].map(w).fillna(0),
                 hit=d["outcome"].isin(["1B", "2B", "3B", "HR"]).astype(int))
    tb = b.groupby(["game_pk", "batter"])["tb"].sum().rename("v").reset_index()
    hits = b.groupby(["game_pk", "batter"])["hit"].sum().rename("v").reset_index()
    pool = {"strikeouts": k, "outs recorded": outs,
            "total bases": tb, "hits": hits}

    out = {}
    for market, g in led.groupby("market"):
        ref = pool.get(market)
        if ref is None:
            continue
        rates = []
        for line, gg in g.groupby("line"):
            over = float((ref["v"] > line).mean())
            n_over = int((gg["side"] == "over").sum())
            rates += [over] * n_over + [1.0 - over] * (len(gg) - n_over)
        if rates:
            out[market] = float(np.mean(rates))
    return out


def _summary(df, label):
    settled = df[df["result"].notna() & (df["result"] != 0.5)]
    push = int((df["result"] == 0.5).sum())
    if not len(settled):
        return f"  {label:<14} no settled picks"
    wr = settled["result"].mean() * 100
    conf = settled[settled["p_side"] >= CONFIDENT]
    line = (f"  {label:<14} {len(settled):>4} picks  {wr:>5.1f}%"
            f"   ({push} push)")
    if len(conf):
        line += (f"   |  >={int(CONFIDENT * 100)}%: {len(conf):>4} picks  "
                 f"{conf['result'].mean() * 100:.1f}%")
    else:
        line += f"   |  >={int(CONFIDENT * 100)}%: none"
    return line


def report(led=None, emit=print):
    if led is None:
        led = grade()
    if led is None or not len(led):
        emit("  no picks recorded yet")
        return
    led = led.copy()
    led["game_date"] = pd.to_datetime(led["game_date"]).dt.date
    today = pd.Timestamp.now().date()
    y = led[led["game_date"] == today - pd.Timedelta(days=1)]
    w = led[led["game_date"] >= today - pd.Timedelta(days=7)]

    emit("\n  === model pick record ===")
    emit(_summary(y, "yesterday"))
    emit(_summary(w, "last 7 days"))
    emit(_summary(led, "all time"))

    settled = led[led["result"].notna() & (led["result"] != 0.5)]
    if len(settled) >= 20:
        emit("\n  by market (all time):")
        for m, g in settled.groupby("market"):
            conf = g[g["p_side"] >= CONFIDENT]
            c = (f"  >={int(CONFIDENT * 100)}%: {len(conf)} at "
                 f"{conf['result'].mean() * 100:.1f}%") if len(conf) else ""
            emit(f"    {m:<15} {len(g):>4} picks  "
                 f"{g['result'].mean() * 100:>5.1f}%{c}")
        try:
            bl = blind_rate(settled)
            emit("\n  vs betting the same side blindly:")
            for m, g in settled.groupby("market"):
                if m not in bl:
                    continue
                got = g["result"].mean() * 100
                base = bl[m] * 100
                emit(f"    {m:<15} model {got:5.1f}%   blind {base:5.1f}%   "
                     f"{got - base:+5.1f}pp")
            emit("  (this, not 52.4%, is the benchmark: a market where the")
            emit("   model always takes one side looks good on base rate alone)")
        except Exception as exc:
            emit(f"  (blind base rate unavailable: {type(exc).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report()


if __name__ == "__main__":
    main()
