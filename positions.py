"""Who can play where, and what a substitution actually costs.

A pinch hitter is not a free swap. He takes a lineup slot that belongs to a
fielder, so either he covers that position himself or somebody moves and a
different glove enters the game. Ignoring that makes every pinch hit look
cheaper than it is: the model would happily hit for the shortstop in the sixth
and leave nobody to play short.

Eligibility is derived from who actually appeared at each position, using the
`fielder_2` through `fielder_9` alignment already captured on every plate
appearance. That is better than a roster designation because it reflects how a
player was genuinely used -- a nominal shortstop who has not played there in two
months is not a shortstop tonight.

Position numbers follow the standard scorekeeping convention: 2 catcher,
3 first base, 4 second, 5 third, 6 short, 7 left, 8 centre, 9 right. The
designated hitter has no fielding position and is therefore the one lineup slot
that can be hit for at no defensive cost.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

POS_NAMES = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
             7: "LF", 8: "CF", 9: "RF"}
POS_PATH = os.path.join(config.PROCESSED_DIR, "position_eligibility.parquet")

# Defensive difficulty, roughly the standard positional adjustment in runs per
# season. Used to price a substitution that pushes someone out of position.
POSITION_VALUE = {2: 12.5, 6: 7.5, 4: 2.5, 5: 2.5, 8: 2.5,
                  7: -7.5, 9: -7.5, 3: -12.5}


def build_eligibility(pa, seasons=None, min_games=3):
    """Games played at each position, per player per season."""
    d = pa[pa["is_pa"]]
    if seasons:
        d = d[d["season"].isin(seasons)]

    frames = []
    for pos in range(2, 10):
        col = f"fielder_{pos}"
        if col not in d.columns:
            continue
        sub = (d[["season", "game_pk", col]].dropna()
               .drop_duplicates(["season", "game_pk", col]))
        sub = sub.rename(columns={col: "player_id"})
        sub["position"] = pos
        frames.append(sub)

    if not frames:
        raise SystemExit("no fielder_* columns; rebuild pa_events")

    stacked = pd.concat(frames, ignore_index=True)
    stacked["player_id"] = stacked["player_id"].astype(int)
    out = (stacked.groupby(["season", "player_id", "position"])
           .size().rename("games").reset_index())
    out = out[out["games"] >= min_games]

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out.to_parquet(POS_PATH, index=False)
    print(f"position eligibility: {len(out):,} player-season-position rows, "
          f"{out['player_id'].nunique():,} players -> {POS_PATH}")
    return out


def load_eligibility():
    if not os.path.exists(POS_PATH):
        raise SystemExit(f"no table at {POS_PATH}; run build_eligibility first")
    return pd.read_parquet(POS_PATH)


class Positions:
    """Eligibility lookups as of a season, using prior seasons plus this one."""

    def __init__(self, elig, season, lookback=2):
        window = elig[elig["season"].between(season - lookback, season)]
        self.by_player = {}
        for pid, grp in window.groupby("player_id"):
            self.by_player[int(pid)] = dict(
                zip(grp["position"], grp["games"]))

    def can_play(self, player_id, position, min_games=3):
        return self.by_player.get(int(player_id), {}).get(position, 0) >= min_games

    def positions(self, player_id):
        return sorted(self.by_player.get(int(player_id), {}).keys())

    def primary(self, player_id):
        p = self.by_player.get(int(player_id))
        return max(p, key=p.get) if p else None

    def describe(self, player_id):
        p = self.by_player.get(int(player_id), {})
        if not p:
            return "no fielding record (likely a designated hitter)"
        return ", ".join(f"{POS_NAMES[k]}({v})"
                         for k, v in sorted(p.items(), key=lambda x: -x[1]))


def lineup_positions(pa, game_pk, team):
    """Which lineup slot held which fielding position in a given game.

    Maps each defensive position to the batting slot of the man playing it, so
    a substitution can be checked against the slot it would vacate.
    """
    d = pa[(pa["game_pk"] == game_pk) & pa["is_pa"]
           & (pa["batting_team"] == team)]
    slot_of = (d.drop_duplicates("lineup_slot")
               .set_index("batter")["lineup_slot"].to_dict())

    f = pa[(pa["game_pk"] == game_pk) & pa["is_pa"]
           & (pa["fielding_team"] == team)]
    out = {}
    for pos in range(2, 10):
        col = f"fielder_{pos}"
        if col not in f.columns or f[col].dropna().empty:
            continue
        pid = int(f[col].dropna().mode().iloc[0])
        if pid in slot_of:
            out[pos] = {"slot": int(slot_of[pid]), "player_id": pid}
    return out


def substitution_cost(positions, defense, incoming_id, outgoing_id, position):
    """What a swap costs on defence, in runs per defensive plate appearance.

    Returns None when the incoming player cannot cover the position at all --
    which makes the substitution illegal without a further move, not merely
    expensive.
    """
    if position is None:
        return 0.0                      # the designated hitter fields nowhere
    if not positions.can_play(incoming_id, position):
        return None
    return defense.get(incoming_id) - defense.get(outgoing_id)


def main():
    import outcome_model as om
    import player_ratings as pr

    pa = om.load_pa()
    elig = build_eligibility(pa)

    print("\nposition coverage by season:")
    print(elig.groupby("season")["player_id"].nunique().to_string())

    pos = Positions(elig, 2025)
    multi = {p: len(v) for p, v in pos.by_player.items()}
    s = pd.Series(multi)
    print(f"\nplayers with fielding records as of 2025: {len(s):,}")
    print(f"  positions per player: mean {s.mean():.2f}, "
          f"{(s == 1).mean() * 100:.0f}% single-position, "
          f"{(s >= 3).mean() * 100:.0f}% cover three or more")

    print("\nmost versatile players (2025 window):")
    for pid in s.sort_values(ascending=False).head(5).index:
        print(f"  {pid}: {pos.describe(pid)}")

    print("\nhow often is the designated hitter slot available to hit for?")
    d = pa[pa["is_pa"] & (pa["season"] == 2025)]
    bat_ids = set(d["batter"].unique())
    no_glove = [b for b in bat_ids if not pos.by_player.get(int(b))]
    print(f"  {len(no_glove):,} of {len(bat_ids):,} batters have no fielding "
          f"record -- these are the free swaps")

    # A worked check of the constraint the decision layer needs.
    game = d["game_pk"].iloc[0]
    team = d[d["game_pk"] == game]["home_team"].iloc[0]
    lp = lineup_positions(pa, game, team)
    print(f"\nexample lineup, game {game}, {team}:")
    for p, info in sorted(lp.items()):
        print(f"  slot {info['slot']}: {POS_NAMES[p]:<3} "
              f"player {info['player_id']}")
    covered = set(lp.keys())
    print(f"  slots with no fielding position (free to pinch hit): "
          f"{9 - len(covered)}")


if __name__ == "__main__":
    main()
