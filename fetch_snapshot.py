"""Build data/*.csv for the app from the public sportsdataverse ESPN box-score release
(hosted on GitHub, so it works from GitHub Actions) and ESPN rosters for heights."""
import io
import pathlib
import numpy as np
import pandas as pd
import requests

import data

URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{y}.csv"
ABBR = {"NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}


def season_year(today=None):
    t = today or pd.Timestamp.today()
    return t.year + 1 if t.month >= 10 else t.year  # ESPN labels 2026-27 as 2027


def fetch_box(y):
    r = requests.get(URL.format(y=y), timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)


def build(raw):
    b = raw[(raw.season_type == 2) & (raw.did_not_play != True) & (raw.minutes > 0)].copy()
    for c in ("team_abbreviation", "opponent_team_abbreviation"):
        b[c] = b[c].replace(ABBR)
    logs = pd.DataFrame({
        "game_id": b.game_id, "date": pd.to_datetime(b.game_date), "player_id": b.athlete_id,
        "player": b.athlete_display_name, "team": b.team_abbreviation,
        "opp": b.opponent_team_abbreviation, "home": b.home_away == "home", "min": b.minutes,
        "pts": b.points, "reb": b.rebounds, "ast": b.assists, "fg3m": b.three_point_field_goals_made,
        "stl": b.steals, "blk": b.blocks, "tov": b.turnovers, "fgm": b.field_goals_made,
        "fga": b.field_goals_attempted, "fg3a": b.three_point_field_goals_attempted,
        "ftm": b.free_throws_made, "fta": b.free_throws_attempted, "pf": b.fouls,
        "plus_minus": b.plus_minus, "oreb": b.offensive_rebounds, "dreb": b.defensive_rebounds})
    return logs.reset_index(drop=True)


def advanced(logs):
    g = logs.groupby(["game_id", "team", "opp"]).agg(
        pts=("pts", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), fgm=("fgm", "sum"),
        oreb=("oreb", "sum"), dreb=("dreb", "sum"), reb=("reb", "sum"), tov=("tov", "sum"),
        tmin=("min", "sum")).reset_index()
    g["poss"] = g.fga + 0.44 * g.fta - g.oreb + g.tov
    o = g.rename(columns=lambda c: "o_" + c if c not in ("game_id",) else c)
    m = g.merge(o, left_on=["game_id", "opp"], right_on=["game_id", "o_team"])
    m["poss_avg"] = (m.poss + m.o_poss) / 2
    t = m.groupby("team").agg(pts=("pts", "sum"), opp_pts=("o_pts", "sum"), poss=("poss_avg", "sum"),
                              n=("game_id", "size"))
    team_adv = pd.DataFrame({"team": t.index, "off_rating": (100 * t.pts / t.poss).values,
                             "def_rating": (100 * t.opp_pts / t.poss).values,
                             "pace": (t.poss / t.n).values})
    team_adv["net_rating"] = team_adv.off_rating - team_adv.def_rating

    p = logs.merge(m[["game_id", "team", "poss_avg", "fga", "fta", "tov", "fgm", "tmin", "reb",
                      "o_reb", "o_fga", "o_fta", "o_fgm", "o_tov", "o_oreb", "o_dreb", "o_pts", "pts",
                      "oreb", "dreb"]].add_suffix("_t").rename(columns={"game_id_t": "game_id",
                                                                         "team_t": "team"}),
                   on=["game_id", "team"])
    p["usg_num"] = (p.fga + .44 * p.fta + p.tov) * (p.tmin_t / 5)
    p["usg_den"] = p["min"] * (p.fga_t + .44 * p.fta_t + p.tov_t)
    p["ast_den"] = (p["min"] / (p.tmin_t / 5)) * p.fgm_t - p.fgm
    p["reb_num"] = p.reb * (p.tmin_t / 5)
    p["reb_den"] = p["min"] * (p.reb_t + p.o_reb_t)
    p["pie_num"] = (p.pts + p.fgm + p.ftm - p.fga - p.fta + p.dreb + p.oreb / 2 + p.ast + p.stl
                    + p.blk / 2 - p.pf - p.tov)
    p["pie_den"] = p.groupby("game_id").pie_num.transform("sum")
    s = p.groupby("player_id").agg(gp=("game_id", "size"), min=("min", "mean"), pts=("pts", "sum"),
                                   fga=("fga", "sum"), fta=("fta", "sum"), ast=("ast", "sum"),
                                   un=("usg_num", "sum"), ud=("usg_den", "sum"), ad=("ast_den", "sum"),
                                   rn=("reb_num", "sum"), rd=("reb_den", "sum"),
                                   pn=("pie_num", "sum"), pd_=("pie_den", "sum"))
    player_adv = pd.DataFrame({
        "player_id": s.index, "gp": s.gp, "min": s["min"],
        "usg_pct": (s.un / s.ud).values, "ts_pct": (s.pts / (2 * (s.fga + .44 * s.fta))).values,
        "ast_pct": (s.ast / s.ad.where(s.ad > 0)).values, "reb_pct": (s.rn / s.rd).values,
        "pie": (s.pn / s.pd_).values})
    return team_adv, player_adv.fillna(0)


def heights(logs):
    """ESPN roster heights (inches). Returns NaN heights if ESPN is unreachable."""
    out = {}
    try:
        teams = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams",
                             timeout=30).json()["sports"][0]["leagues"][0]["teams"]
        for t in teams:
            r = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/"
                             f"{t['team']['id']}/roster", timeout=30).json()
            for a in r.get("athletes", []):
                if a.get("height"):
                    out[int(a["id"])] = float(a["height"])
    except Exception as e:
        print("height fetch failed:", type(e).__name__, e)
    ids = logs.player_id.unique()
    return pd.DataFrame({"player_id": ids, "height_in": [out.get(int(i), np.nan) for i in ids]})


if __name__ == "__main__":
    y = season_year()
    logs = build(fetch_box(y))
    team_adv, player_adv = advanced(logs)
    bio = heights(logs)
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    logs.to_csv(out / "logs.csv", index=False)
    bio.to_csv(out / "bio.csv", index=False)
    team_adv.to_csv(out / "team_adv.csv", index=False)
    player_adv.to_csv(out / "player_adv.csv", index=False)
    print(f"saved {len(logs)} player-games through {logs.date.max():%Y-%m-%d}; "
          f"heights for {bio.height_in.notna().sum()}/{len(bio)} players")
