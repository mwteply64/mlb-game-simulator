"""Predict a lineup that has not been posted yet.

Roughly one player in seven on a Yahoo slate belongs to a club that has not
posted its card by the time contests lock. Dropping those games leaves real
value unpriced, and it leaves it unpriced in exactly the spot where the rest of
the field is also blind -- most entrants simply avoid unposted games. Estimating
where others abstain is one of the few situations in this project where a model
can beat the field rather than merely match it.

**This predicts, it does not optimise.** The lineup optimiser from the sibling
project answers "what should this manager do"; that is the wrong question here
and would bias every unposted team upward, since real lineups are worse than
optimal ones. What actually predicts tomorrow's card is recent usage:

* the club's most recent lineup is the single strongest signal,
* split by the handedness of the opposing starter, because clubs platoon,
* filtered against the active roster so anyone since placed on the IL drops,
* with vacancies filled by whoever has started most often at that spot.

Rather than commit to one guess, `candidates()` returns several recent lineups
with weights, so the simulator can average across them and the resulting
projections widen to reflect that we are guessing. A single guessed lineup would
produce projections as confident as a posted one, which is the error worth
avoiding.

    python projected_lineups.py --team NYY --vs-hand R
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

LOOKBACK_GAMES = 12
DECAY = 0.75          # per game of age; the last card dominates but is not alone


def _lineup_frame(pa, season=None):
    d = pa[pa["is_pa"] & pa["lineup_slot"].notna()].copy()
    if season:
        d = d[d["season"] == season]
    d["game_date"] = pd.to_datetime(d["game_date"])
    # The batter who took the slot's first plate appearance is the starter
    # there; later occupants are substitutes and must not be predicted.
    d = d.sort_values(["game_pk", "at_bat_number"])
    first = d.drop_duplicates(subset=["game_pk", "batting_team", "lineup_slot"],
                              keep="first")
    return first[["game_pk", "game_date", "batting_team", "lineup_slot",
                  "batter", "p_throws"]]


def candidates(pa, team, vs_hand=None, asof=None, season=None,
               n_candidates=3, active=None):
    """Plausible lineups for a club, most recent first, with weights.

    Weights decay with age, so the last card carries the most mass without the
    others being ignored. When `vs_hand` is given, cards against that hand are
    preferred -- a club's lineup against a left-hander is a better guide to its
    next lineup against a left-hander than its last card overall.
    """
    lf = _lineup_frame(pa, season)
    lf = lf[lf["batting_team"] == team]
    if asof is not None:
        lf = lf[lf["game_date"] < pd.Timestamp(asof)]
    if not len(lf):
        return []

    games = (lf.groupby(["game_pk", "game_date"])["p_throws"]
             .first().reset_index().sort_values("game_date", ascending=False))
    games = games.head(LOOKBACK_GAMES)
    if vs_hand:
        same = games[games["p_throws"] == vs_hand]
        if len(same) >= 2:
            games = same

    out = []
    for rank, row in enumerate(games.itertuples(index=False)):
        card = lf[lf["game_pk"] == row.game_pk].sort_values("lineup_slot")
        ids = [int(x) for x in card["batter"]]
        if len(ids) != 9:
            continue
        if active is not None and any(i not in active for i in ids):
            # A card containing someone no longer on the roster cannot be
            # tonight's lineup. Repairing it would invent a lineup nobody has
            # used; skipping and falling through to an older card keeps every
            # candidate something the manager has actually written down.
            continue
        out.append({"game_pk": int(row.game_pk), "date": row.game_date,
                    "vs_hand": row.p_throws, "lineup": ids,
                    "weight": DECAY ** rank})
        if len(out) >= n_candidates:
            break

    if not out:
        return []
    tot = sum(o["weight"] for o in out)
    for o in out:
        o["weight"] /= tot
    return out


def most_likely(pa, team, vs_hand=None, asof=None, season=None, active=None):
    """Single best guess, for callers that cannot average over candidates."""
    c = candidates(pa, team, vs_hand, asof, season, 1, active)
    return c[0]["lineup"] if c else None


def resolve(pa, team, posted_ids=None, vs_hand=None, asof=None, season=None,
            active=None, n_candidates=3):
    """The lineup to simulate, and where it came from.

    **A projection is strictly a fallback.** If a real lineup has been posted it
    is used unchanged and no estimate is consulted -- a guessed card must never
    displace a known one, however good the estimator looks. `posted_ids` being
    a full nine is the only test that matters; anything shorter means the club
    has not posted and the fallback applies.

    Returns (candidates, source) where source is "posted" or "projected", so
    every downstream consumer can mark which projections rest on a guess.
    """
    if posted_ids and len([p for p in posted_ids if p]) == 9:
        return ([{"lineup": [int(x) for x in posted_ids], "weight": 1.0,
                  "date": None, "vs_hand": vs_hand, "game_pk": None}],
                "posted")
    cand = candidates(pa, team, vs_hand, asof, season, n_candidates, active)
    return cand, ("projected" if cand else "unavailable")


def slot_stability(pa, season):
    """How repeatable each batting-order slot is, one slot at a time.

    Aggregate repeatability understates what matters here. The top of the order
    is where the expensive hitters bat and where plate appearances are
    concentrated, and it is set far more rigidly than the bottom, which absorbs
    the platoon and rest decisions. If slots 1-4 hold while 7-9 churn, a
    projected lineup is far more useful than the overall figure suggests.
    """
    lf = _lineup_frame(pa, season)
    rows = []
    for team, g in lf.groupby("batting_team"):
        order = (g.groupby(["game_pk", "game_date"]).size().reset_index()
                 .sort_values("game_date"))
        prev = None
        for row in order.itertuples(index=False):
            card = (g[g["game_pk"] == row.game_pk].sort_values("lineup_slot")
                    [["lineup_slot", "batter"]])
            cur = {int(r.lineup_slot): int(r.batter)
                   for r in card.itertuples(index=False)}
            if prev and len(cur) == 9 and len(prev) == 9:
                for slot in range(1, 10):
                    rows.append({"slot": slot,
                                 "same": cur.get(slot) == prev.get(slot),
                                 "still_in": cur.get(slot) in set(prev.values())})
            prev = cur
    return pd.DataFrame(rows)


def coverage(pa, season, asof=None):
    """How often the last card is exactly repeated -- the ceiling on this idea.

    If clubs rarely repeat a lineup verbatim, predicting one is hopeless and the
    honest move is to widen the projections a great deal or skip those games.
    This measures it rather than assuming.
    """
    lf = _lineup_frame(pa, season)
    rows = []
    for team, g in lf.groupby("batting_team"):
        games = (g.groupby(["game_pk", "game_date"]).size().reset_index()
                 .sort_values("game_date"))
        prev = None
        for row in games.itertuples(index=False):
            card = tuple(g[g["game_pk"] == row.game_pk]
                         .sort_values("lineup_slot")["batter"].astype(int))
            if len(card) == 9 and prev is not None and len(prev) == 9:
                same = sum(1 for a, b in zip(card, prev) if a == b)
                rows.append({"team": team, "exact": card == prev,
                             "same_players": len(set(card) & set(prev)),
                             "same_slot": same})
            prev = card
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--team")
    ap.add_argument("--vs-hand")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()

    import outcome_model as om
    pa = om.load_pa()

    if args.coverage or not args.team:
        c = coverage(pa, args.season)
        print(f"consecutive-lineup similarity, {args.season} ({len(c):,} pairs)")
        print(f"  identical card, same order : {c['exact'].mean() * 100:.1f}%")
        print(f"  same nine players          : "
              f"{(c['same_players'] == 9).mean() * 100:.1f}%")
        print(f"  mean players in same slot  : {c['same_slot'].mean():.2f} of 9")
        print(f"  mean shared players        : {c['same_players'].mean():.2f} of 9")
        print("\n  -> a projected lineup should be treated as roughly this")
        print("     accurate, and the simulator should average over several")
        print("     candidates rather than trusting one.")
        return

    import active_rosters
    r = active_rosters.load()
    act = r.as_of(args.team, pd.Timestamp.now()) if r.available() else None
    for c in candidates(pa, args.team, args.vs_hand, season=args.season,
                        active=act):
        print(f"  {c['date'].date()}  vs {c['vs_hand']}HP  "
              f"weight {c['weight']:.2f}  {c['lineup']}")


if __name__ == "__main__":
    main()
