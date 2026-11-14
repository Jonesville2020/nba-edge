import os
"""Data loading: nba_api for results, The Odds API for lines, demo fallback."""
import math
import numpy as np
import pandas as pd
import requests

ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def current_season(today=None) -> str:
    t = today or pd.Timestamp.today()
    y = t.year if t.month >= 10 else t.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def load_games(season: str | None = None) -> pd.DataFrame:
    """Completed games with columns date, home, away, margin."""
    from nba_api.stats.endpoints import leaguegamelog
    log = leaguegamelog.LeagueGameLog(
        season=season or current_season(), timeout=30).get_data_frames()[0]
    log = log[log.MATCHUP.str.contains("vs.")]  # home rows only
    away = log.MATCHUP.str.split(" vs. ").str[1]
    abbr_to_name = dict(zip(log.TEAM_ABBREVIATION, log.TEAM_NAME))
    full = _full_names()
    out = pd.DataFrame({
        "date": pd.to_datetime(log.GAME_DATE),
        "home": log.TEAM_ABBREVIATION.map(full),
        "away": away.map(full),
        "margin": log.PLUS_MINUS.astype(float),
    })
    return out.dropna().reset_index(drop=True)


def _full_names() -> dict:
    from nba_api.stats.static import teams
    return {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}


def load_odds(api_key: str) -> pd.DataFrame:
    """Best-available (consensus median) lines per game from The Odds API."""
    r = requests.get(ODDS_URL, params=dict(
        apiKey=api_key, regions="us", markets="h2h,spreads", oddsFormat="american"),
        timeout=30)
    r.raise_for_status()
    rows = []
    for g in r.json():
        h, a = g["home_team"], g["away_team"]
        ml_h, ml_a, sp, spp = [], [], [], []
        for bk in g["bookmakers"]:
            for m in bk["markets"]:
                oc = {o["name"]: o for o in m["outcomes"]}
                if m["key"] == "h2h" and h in oc and a in oc:
                    ml_h.append(oc[h]["price"]); ml_a.append(oc[a]["price"])
                if m["key"] == "spreads" and h in oc:
                    sp.append(oc[h]["point"]); spp.append(oc[h]["price"])
        if ml_h:
            rows.append(dict(home=h, away=a, home_ml=float(np.median(ml_h)),
                             away_ml=float(np.median(ml_a)),
                             home_spread=float(np.median(sp)) if sp else np.nan,
                             spread_price=float(np.median(spp)) if spp else -110,
                             commence=g["commence_time"]))
    return pd.DataFrame(rows)


def demo_games(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic season + slate so the app runs offline."""
    rng = np.random.default_rng(seed)
    names = sorted(_full_names().values()) if _can_names() else [f"Team {i}" for i in range(30)]
    true = dict(zip(names, rng.normal(0, 4, len(names))))
    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=90)
    for d in range(90):
        picks = rng.permutation(names)
        for i in range(0, 20, 2):
            h, a = picks[i], picks[i + 1]
            rows.append(dict(date=start + pd.Timedelta(days=d), home=h, away=a,
                             margin=true[h] - true[a] + 2.5 + rng.normal(0, 12)))
    games = pd.DataFrame(rows)
    slate = []
    p = rng.permutation(names)
    for i in range(0, 12, 2):
        h, a = p[i], p[i + 1]
        m = true[h] - true[a] + 2.5 + rng.normal(0, 2)
        pw = 0.5 * (1 + math.erf(m / 12 / 2 ** 0.5))
        ml = lambda q: -100 * q / (1 - q) if q > .5 else 100 * (1 - q) / q
        pw = min(max(pw, .05), .95)
        slate.append(dict(home=h, away=a, home_ml=round(ml(pw + .02)), away_ml=round(ml(1 - pw + .02)),
                          home_spread=round(-m * 2) / 2, spread_price=-110))
    return games, pd.DataFrame(slate)


def _can_names() -> bool:
    try:
        _full_names()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- player data
def _abbr_by_name() -> dict:
    from nba_api.stats.static import teams
    return {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}


def team_names() -> dict:
    """abbreviation -> full name"""
    return _full_names()


def load_player_data(season: str | None = None):
    """Returns (logs, bio, team_adv, player_adv) for this season only."""
    from nba_api.stats.endpoints import (leaguedashplayerbiostats, leaguedashplayerstats,
                                         leaguedashteamstats, playergamelogs)
    season = season or current_season()
    proxy = os.getenv("NBA_PROXY") or None
    try:  # stats.nba.com silently drops many cloud IPs; fail fast instead of hanging
        requests.get("https://stats.nba.com", timeout=8, proxies={"https": proxy} if proxy else None)
    except Exception as e:
        raise RuntimeError("stats.nba.com is not reachable from this server (cloud IPs are often "
                           f"blocked). Set an NBA_PROXY secret or run locally. ({type(e).__name__})")
    raw = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=25, proxy=proxy).get_data_frames()[0]
    logs = pd.DataFrame({
        "game_id": raw.GAME_ID, "date": pd.to_datetime(raw.GAME_DATE),
        "player_id": raw.PLAYER_ID, "player": raw.PLAYER_NAME, "team": raw.TEAM_ABBREVIATION,
        "opp": raw.MATCHUP.str[-3:], "home": raw.MATCHUP.str.contains("vs."),
        "min": raw.MIN.astype(float), "pts": raw.PTS, "reb": raw.REB, "ast": raw.AST,
        "fg3m": raw.FG3M, "stl": raw.STL, "blk": raw.BLK, "tov": raw.TOV,
        "fgm": raw.FGM, "fga": raw.FGA, "fg3a": raw.FG3A, "ftm": raw.FTM, "fta": raw.FTA,
        "pf": raw.PF, "plus_minus": raw.PLUS_MINUS})
    b = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season, timeout=25, proxy=proxy).get_data_frames()[0]
    bio = pd.DataFrame({"player_id": b.PLAYER_ID, "height_in": b.PLAYER_HEIGHT_INCHES})
    t = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=25, proxy=proxy).get_data_frames()[0]
    ab = _abbr_by_name()
    team_adv = pd.DataFrame({"team": t.TEAM_NAME.map(ab), "off_rating": t.OFF_RATING,
                             "def_rating": t.DEF_RATING, "net_rating": t.NET_RATING, "pace": t.PACE})
    pa = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=25, proxy=proxy).get_data_frames()[0]
    player_adv = pa[["PLAYER_ID", "GP", "MIN", "OFF_RATING", "DEF_RATING", "NET_RATING",
                     "USG_PCT", "TS_PCT", "AST_PCT", "REB_PCT", "PIE"]].rename(
        columns=str.lower).rename(columns={"player_id": "player_id"})
    return logs, bio, team_adv, player_adv


def demo_player_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    abbrs = sorted(_full_names())
    team_def = dict(zip(abbrs, rng.normal(0, 1, len(abbrs))))
    team_off = dict(zip(abbrs, rng.normal(0, 1, len(abbrs))))
    team_pace = dict(zip(abbrs, rng.normal(0, 1, len(abbrs))))
    mins = [34, 32, 30, 28, 26, 22, 18, 14, 10, 6]
    players, bio = [], []
    pid = 1
    for t in abbrs:
        tall = rng.normal(0, 1.2)
        for i in range(10):
            h = float(np.clip(rng.normal(79 + tall, 3), 72, 88))
            skill = rng.uniform(0.6, 1.4)
            players.append(dict(player_id=pid, name=f"{t} Player {i + 1}", team=t, mp=mins[i],
                                skill=skill, h=h))
            bio.append(dict(player_id=pid, height_in=h))
            pid += 1
    pl = pd.DataFrame(players)
    th = pl.groupby("team").h.mean()
    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=60)
    gid = 0
    for d in range(60):
        order = rng.permutation(abbrs)
        for i in range(0, 20, 2):
            home, away = order[i], order[i + 1]
            gid += 1
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                zh = (th[opp] - th.mean()) / th.std()
                for p in pl[pl.team == team].itertuples():
                    if rng.random() < 0.06:
                        continue
                    m = max(2, rng.normal(p.mp, 3))
                    pf = rng.poisson(2.2 * m / 30)
                    if pf >= 5:
                        m *= 0.7
                    rate = m / 36 * p.skill * (1 - 0.04 * team_def[opp]) * (1 + 0.02 * team_pace[opp])
                    big = (p.h - 79) / 3
                    pts = rng.poisson(max(0.5, 16 * rate * (1.03 if is_home else 0.97)))
                    reb = rng.poisson(max(0.3, 5.5 * rate * (1 + 0.15 * big) * (1 - 0.05 * zh)))
                    ast = rng.poisson(max(0.3, 4 * rate * (1 - 0.15 * big)))
                    fg3a = rng.poisson(max(.2, 5 * rate))
                    fga = fg3a + rng.poisson(max(.5, 8 * rate))
                    fta = rng.poisson(max(.3, 3.5 * rate))
                    rows.append(dict(game_id=gid, fgm=rng.binomial(fga, .46), fga=fga, fg3a=fg3a,
                                     ftm=rng.binomial(fta, .78), fta=fta, date=start + pd.Timedelta(days=d), player_id=p.player_id,
                                     player=p.name, team=team, opp=opp, home=is_home, min=m, pts=pts,
                                     reb=reb, ast=ast, fg3m=rng.poisson(max(.1, 1.8 * rate)),
                                     stl=rng.poisson(.9 * rate), blk=rng.poisson(.6 * rate),
                                     tov=rng.poisson(1.8 * rate), pf=pf, plus_minus=rng.normal(0, 8)))
    logs = pd.DataFrame(rows)
    team_adv = pd.DataFrame({"team": abbrs, "off_rating": [112 + 3 * team_off[t] for t in abbrs],
                             "def_rating": [112 + 3 * team_def[t] for t in abbrs],
                             "net_rating": [3 * (team_off[t] - team_def[t]) for t in abbrs],
                             "pace": [99 + 2 * team_pace[t] for t in abbrs]})
    player_adv = pd.DataFrame({"player_id": pl.player_id, "gp": 20, "min": pl.mp,
                               "off_rating": 112, "def_rating": 112, "net_rating": 0.0,
                               "usg_pct": rng.uniform(.12, .30, len(pl)),
                               "ts_pct": rng.uniform(.5, .64, len(pl)),
                               "ast_pct": rng.uniform(.05, .35, len(pl)),
                               "reb_pct": rng.uniform(.04, .2, len(pl)),
                               "pie": rng.uniform(.06, .18, len(pl))})
    return logs, pd.DataFrame(bio), team_adv, player_adv
