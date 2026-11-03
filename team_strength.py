"""Inject Elo-derived team strength into the simulator's talent layer.

The simulator's own team-strength estimate is built bottom-up: project each
player, then add them together. Measured against 2025, that estimate orders
teams worse than a rating fit purely to game results -- Spearman 0.8005 against
Elo's 0.8855 -- and Spearman is immune to scale, so this is an accuracy gap
rather than a compression artifact.

Blending the two at the *output* level (averaging win probabilities) only helps
the moneyline. Injecting at the *talent* level lets the simulation inherit Elo's
ordering while still producing a full run distribution, so totals and every
derived market benefit too.

The correction is expressed in runs per game and applied to the outcome vectors
the same way park factors, home advantage and team defense already are.

Nothing here uses information from the future: Elo ratings walk forward and see
only prior games, and the blend weights are fitted on seasons before the one
being predicted.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config

OUTCOMES = config.OUTCOMES

ELO_START = 1500.0
ELO_K = 4.0
ELO_HOME_ADV = 24.0
ELO_SEASON_REGRESS = 0.25


def elo_by_game(pa, k=ELO_K, home_adv=ELO_HOME_ADV,
                regress=ELO_SEASON_REGRESS):
    """Pre-game Elo ratings for every game, walking the history forward.

    Returns one row per game with each side's rating as it stood *before* that
    game was played.
    """
    games = (pa[pa["is_pa"]].groupby("game_pk")
             .agg(season=("season", "first"), date=("game_date", "first"),
                  home=("home_team", "first"), away=("away_team", "first"))
             .reset_index())
    runs = (pa[pa["is_pa"]].groupby(["game_pk", "batting_team"])["runs_pa"]
            .sum().reset_index()
            .pivot(index="game_pk", columns="batting_team", values="runs_pa"))

    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    rating, prev_season, rows = {}, None, []
    for r in games.itertuples(index=False):
        if prev_season is not None and r.season != prev_season:
            for t in rating:
                rating[t] = ELO_START + (rating[t] - ELO_START) * (1 - regress)
        prev_season = r.season

        rh = rating.setdefault(r.home, ELO_START)
        ra = rating.setdefault(r.away, ELO_START)
        rows.append({"game_pk": r.game_pk, "season": r.season,
                     "home_team": r.home, "away_team": r.away,
                     "elo_home": rh, "elo_away": ra})

        hr, ar = runs.at[r.game_pk, r.home], runs.at[r.game_pk, r.away]
        if pd.isna(hr) or pd.isna(ar):
            continue
        expected = 1.0 / (1.0 + 10 ** (-(rh + home_adv - ra) / 400.0))
        result = 1.0 if hr > ar else 0.0
        delta = k * np.log(abs(hr - ar) + 1.0) * (result - expected)
        rating[r.home] = rh + delta
        rating[r.away] = ra - delta

    return pd.DataFrame(rows)


def fit_elo_to_runs(pa, elo, seasons):
    """Runs per game implied by one point of Elo rating.

    Fitted rather than assumed: regress each game's actual run differential on
    the pre-game rating difference over prior seasons.
    """
    runs = (pa[pa["is_pa"]].groupby(["game_pk", "batting_team"])["runs_pa"]
            .sum().reset_index())
    e = elo[elo["season"].isin(seasons)].copy()

    h = runs.rename(columns={"batting_team": "home_team", "runs_pa": "hr"})
    a = runs.rename(columns={"batting_team": "away_team", "runs_pa": "ar"})
    e = e.merge(h, on=["game_pk", "home_team"], how="left")
    e = e.merge(a, on=["game_pk", "away_team"], how="left")
    e = e.dropna(subset=["hr", "ar"])

    x = (e["elo_home"] - e["elo_away"]).to_numpy()
    y = (e["hr"] - e["ar"]).to_numpy()
    if len(x) < 100 or x.std() == 0:
        return 0.01
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def team_strength_table(sim_preds, games_meta):
    """The simulator's own implied strength: mean run differential per game."""
    m = sim_preds.merge(games_meta, on="game_pk", how="left")
    m["diff"] = m["sim_home_runs"] - m["sim_away_runs"]
    out = {}
    teams = set(m["home_team"].dropna()) | set(m["away_team"].dropna())
    for t in teams:
        h = m.loc[m["home_team"] == t, "diff"]
        a = m.loc[m["away_team"] == t, "diff"]
        n = len(h) + len(a)
        if n:
            out[t] = (h.sum() - a.sum()) / n
    return pd.Series(out, name="sim_strength")


def fit_blend(sim_strength, elo_strength, actual_strength):
    """Weights for combining the two strength estimates.

    Fitted on standardized inputs against what actually happened, so the blend
    reflects how much each estimate is genuinely worth rather than a guess.
    Returns (w_elo, w_sim) summing to 1.
    """
    df = pd.concat([sim_strength.rename("sim"),
                    elo_strength.rename("elo"),
                    actual_strength.rename("act")], axis=1).dropna()
    if len(df) < 10:
        return 0.7, 0.3

    def z(s):
        return (s - s.mean()) / (s.std() if s.std() else 1.0)

    X = np.column_stack([np.ones(len(df)), z(df["elo"]), z(df["sim"])])
    beta = np.linalg.lstsq(X, df["act"].to_numpy(), rcond=None)[0]
    w_elo, w_sim = max(beta[1], 0.0), max(beta[2], 0.0)
    total = w_elo + w_sim
    if total <= 0:
        return 0.7, 0.3
    return w_elo / total, w_sim / total


def runs_adjustment(elo_home, elo_away, sim_home_strength, sim_away_strength,
                    runs_per_elo, w_elo, strength=1.0):
    """Per-team run-per-game correction to apply to the talent layer.

    Each team's blended strength is compared with what the simulator already
    believes, and the shortfall becomes the adjustment. `strength` scales the
    whole correction so it can be dialled back without refitting.

    Returned as (away, home) to match the simulator's team indexing, where 0 is
    the away side throughout. Returning home-first silently applies each team's
    correction to its opponent.
    """
    elo_h = (elo_home - ELO_START) * runs_per_elo
    elo_a = (elo_away - ELO_START) * runs_per_elo

    target_h = w_elo * elo_h + (1 - w_elo) * sim_home_strength
    target_a = w_elo * elo_a + (1 - w_elo) * sim_away_strength

    return (strength * (target_a - sim_away_strength),
            strength * (target_h - sim_home_strength))


def apply_runs_delta(probs, delta_runs, lw, split=0.5):
    """Shift an outcome vector by a run-per-game amount.

    Half the correction goes to the team's offence and half to its run
    prevention by default -- the simulator has no basis for attributing a
    team-level residual to one side or the other, and splitting it evenly is
    the assumption that adds least.

    Applied to the ways a batter reaches base, with the compensating mass taken
    out of outs, and sized through the empirical linear weights so the run
    effect comes out to the amount asked for.
    """
    if not delta_runs:
        return probs

    reach = [OUTCOMES.index(o) for o in ("BB", "HBP", "1B", "2B", "3B", "HR")]
    i_out = OUTCOMES.index("OUT")

    # Run value moved per unit of probability shifted from outs to reaching.
    per_unit = float(np.mean(lw[reach]) - lw[i_out])
    if abs(per_unit) < 1e-9:
        return probs

    # Roughly 38 plate appearances per team-game.
    shift = (delta_runs * split) / (per_unit * 38.0)

    v = probs.copy()
    mass = v[reach].sum()
    if mass <= 0:
        return probs
    scale = 1.0 + shift / mass
    scale = float(np.clip(scale, 0.5, 1.8))
    v[reach] *= scale
    moved = v[reach].sum() - mass
    v[i_out] = max(v[i_out] - moved, 1e-6)
    return v / v.sum()
