"""Player matchup analysis. Pure pandas, no network. Uses this season's games only."""
import numpy as np
import pandas as pd

import heat as heatmod

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
LABELS = {"pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM", "stl": "STL",
          "blk": "BLK", "tov": "TOV", "pra": "PTS+REB+AST"}
ALL = STATS + ["pra"]


def prep(logs: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """logs: game_id,date,player_id,player,team,opp,home,min,<STATS>,plus_minus.
    Adds height, pra, and the opposing lineup's minutes-weighted height per game."""
    df = logs[logs["min"] > 0].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["pra"] = df.pts + df.reb + df.ast
    df = df.merge(bio[["player_id", "height_in"]], on="player_id", how="left")
    h = df.dropna(subset=["height_in"])
    lineup = (h.assign(hw=h.height_in * h["min"]).groupby(["game_id", "team"])
              .agg(hw=("hw", "sum"), m=("min", "sum")))
    lineup["lineup_h"] = lineup.hw / lineup.m
    lineup = lineup["lineup_h"].reset_index().rename(columns={"team": "opp", "lineup_h": "opp_lineup_h"})
    df = df.merge(lineup, on=["game_id", "opp"], how="left")
    df = df.sort_values(["player_id", "date"])
    df["rest_days"] = df.groupby("player_id").date.diff().dt.days
    df["b2b"] = df.rest_days == 1
    df = _flag_games(df.reset_index(drop=True))
    absn = get_absences(df)
    for who, key in (("tm_out", "team"), ("opp_out", "opp")):
        names = absn.groupby(["game_id", "team"]).player.agg(", ".join).rename(who)
        df = df.join(names, on=["game_id", key]).fillna({who: ""})
    return df


def _flag_games(df: pd.DataFrame) -> pd.DataFrame:
    """Flag games whose box score is distorted by foul trouble, blowouts or early exits."""
    if "pf" not in df:
        df["pf"] = 0
    tp = df.groupby(["game_id", "team"]).pts.sum().rename("team_pts").reset_index()
    opp_pts = tp.rename(columns={"team": "opp", "team_pts": "opp_pts"})
    df = df.merge(tp, on=["game_id", "team"]).merge(opp_pts, on=["game_id", "opp"], how="left")
    df["game_margin"] = df.team_pts - df.opp_pts
    avg = df.groupby("player_id")["min"].transform("mean")
    foul = (df.pf >= 5) | ((df.pf >= 4) & (df["min"] < 0.8 * avg))
    blowout = (df.game_margin.abs() >= 20) & (df["min"] < 0.85 * avg)
    short = (df["min"] < 0.6 * avg) & ~foul & ~blowout
    df["flag"] = np.select([foul, blowout, short],
                           ["Foul trouble", "Blowout (garbage time)", "Short minutes (early exit?)"], "")
    df["abnormal"] = df.flag != ""
    return df


def get_absences(df: pd.DataFrame, min_mpg=20, min_games=5) -> pd.DataFrame:
    """Rotation players (>= min_mpg) who missed a game their team played while on that team."""
    span = df.groupby(["team", "player_id"]).agg(mpg=("min", "mean"), n=("min", "size"),
                                                 first=("date", "min"), last=("date", "max"),
                                                 player=("player", "last")).reset_index()
    rot = span[(span.mpg >= min_mpg) & (span.n >= min_games)]
    tg = df[["team", "game_id", "date"]].drop_duplicates()
    cand = rot.merge(tg, on="team")
    cand = cand[(cand.date >= cand["first"]) & (cand.date <= cand["last"])]
    played = df[["team", "player_id", "game_id"]].assign(_p=1)
    cand = cand.merge(played, on=["team", "player_id", "game_id"], how="left")
    return cand[cand._p.isna()][["game_id", "team", "player_id", "player"]].reset_index(drop=True)


def team_context(df: pd.DataFrame, team_adv: pd.DataFrame) -> pd.DataFrame:
    """Per team: ratings, pace, rotation height, and tier labels (tertiles)."""
    h = df.dropna(subset=["height_in"])
    ht = (h.assign(hw=h.height_in * h["min"]).groupby("team")
          .apply(lambda g: g.hw.sum() / g["min"].sum(), include_groups=False).rename("height"))
    ctx = team_adv.set_index("team").join(ht)

    def tier(s, labels, ascending=True):
        r = s.rank(method="first", ascending=ascending)
        return pd.cut(r, [0, len(s) / 3, 2 * len(s) / 3, len(s)], labels=labels).astype(str)

    ctx["def_tier"] = tier(ctx.def_rating, ["Elite defense", "Average defense", "Weak defense"])
    ctx["off_tier"] = tier(ctx.off_rating, ["Weak offense", "Average offense", "Elite offense"])
    ctx["pace_tier"] = tier(ctx.pace, ["Slow pace", "Average pace", "Fast pace"])
    ctx["height_tier"] = tier(ctx.height, ["Smaller lineup", "Average size", "Bigger lineup"])
    return ctx


def opp_factors(df: pd.DataFrame, min_avg_minutes=15, k=40) -> pd.DataFrame:
    """How much each team allows relative to what each player normally produces
    (1.10 = opponents score/rebound/etc. 10% above their own average against this team)."""
    base = df.groupby("player_id").agg(**{s: (s, "mean") for s in ALL}, mm=("min", "mean"))
    q = df.join(base, on="player_id", rsuffix="_avg")
    q = q[(q.mm >= min_avg_minutes)]
    out = {}
    for s in ALL:
        v = q[s] / q[f"{s}_avg"].replace(0, np.nan)
        g = v.groupby(q.opp)
        n, mean = g.count(), g.mean()
        out[s] = 1 + (mean - 1) * n / (n + k)
    return pd.DataFrame(out)


def _ew_mean(x: pd.Series, half_life=8) -> float:
    w = 0.5 ** (np.arange(len(x))[::-1] / half_life)
    return float(np.average(x, weights=w))


def _split_row(name, g, season_avg):
    row = {"split": name, "games": len(g)}
    for s in ALL:
        row[s] = g[s].mean() if len(g) else np.nan
    return row


def _ratio_rows(df, min_avg_minutes=15):
    base = df.groupby("player_id").agg(**{s: (s, "mean") for s in ALL}, mm=("min", "mean"))
    q = df.join(base, on="player_id", rsuffix="_avg")
    q = q[q.mm >= min_avg_minutes]
    return pd.DataFrame({s: q[s] / q[f"{s}_avg"].replace(0, np.nan) for s in ALL}
                        ).assign(game_id=q.game_id, opp=q.opp)


def _shrunk_ratio(a: pd.Series, b: pd.Series, n: int, k: float) -> float:
    if len(a) == 0 or len(b) == 0 or b.mean() == 0 or np.isnan(b.mean()):
        return 1.0
    return float(1 + (a.mean() / b.mean() - 1) * n / (n + k))


def analyze(df, ctx, factors, absn, player_id, opp, home: bool,
            out_teammates=(), out_opp=(), clean=True, window=5, lean=0.0) -> dict:
    p_all = df[df.player_id == player_id].sort_values("date")
    if p_all.empty:
        return {}
    p = p_all[~p_all.abnormal] if clean and (~p_all.abnormal).sum() >= 5 else p_all
    n_excluded = len(p_all) - len(p)
    season = {s: p[s].mean() for s in ALL}
    last5, last10 = p.tail(5), p.tail(10)
    hf_n = len(p)

    adjust, tm_f, opp_f = _adjustments(df, absn, p, p_all, opp, out_teammates, out_opp)
    heat = heatmod.compute(p, window)
    heat_f = _heat_factors(heat, lean, season)
    proj = []
    for s in ALL:
        base = _ew_mean(p[s])
        of = factors.loc[opp, s] if opp in factors.index and not np.isnan(factors.loc[opp, s]) else 1.0
        side = p[p.home == home][s]
        hf = 1.0
        if len(side) >= 3 and season[s] > 0:
            raw = side.mean() / season[s]
            hf = 1 + (raw - 1) * len(side) / (len(side) + 10)
        est = base * of * hf * tm_f[s] * opp_f[s] * heat_f[s]
        sd = p[s].std(ddof=1) if len(p) > 1 else np.nan
        proj.append({"stat": LABELS[s], "key": s, "season avg": season[s], "last 5": last5[s].mean(),
                     "weighted form": base, "opp factor": of, "home/away factor": hf,
                     "teammates-out factor": tm_f[s], "opp-out factor": opp_f[s], "heat factor": heat_f[s],
                     "projection": est, "low (25%)": max(0, est - 0.674 * sd),
                     "high (75%)": est + 0.674 * sd})
    proj = pd.DataFrame(proj)

    # Splits
    o = ctx.loc[opp] if opp in ctx.index else None
    tm = p.join(ctx[["def_tier", "off_tier", "pace_tier", "height_tier"]], on="opp")
    groups = [("Home", tm[tm.home]), ("Away", tm[~tm.home]),
              ("Back-to-back", tm[tm.b2b]), ("Rested (2+ days)", tm[tm.rest_days >= 2])]
    for col in ["def_tier", "off_tier", "pace_tier", "height_tier"]:
        for label in sorted(tm[col].dropna().unique()):
            groups.append((f"vs {label}", tm[tm[col] == label]))
    groups.append((f"vs {opp} (this season)", tm[tm.opp == opp]))
    splits = pd.DataFrame([_split_row(n, g, season) for n, g in groups])
    splits = splits[splits.games > 0].reset_index(drop=True)

    relevant = {"Home" if home else "Away"}
    if o is not None:
        relevant |= {f"vs {o.def_tier}", f"vs {o.off_tier}", f"vs {o.pace_tier}",
                     f"vs {o.height_tier}", f"vs {opp} (this season)"}
    splits["matches this game"] = splits.split.isin(relevant)

    # Height relationship
    hh = p.dropna(subset=["opp_lineup_h"])
    corr = {}
    if len(hh) >= 8:
        for s in ["pts", "reb", "ast", "pra"]:
            c = hh[s].corr(hh.opp_lineup_h)
            corr[s] = None if np.isnan(c) else float(c)

    patterns = _patterns(p, season, splits, last5, o, opp, corr)
    patterns += _context_notes(p_all, p, opp, n_excluded)
    return dict(player=p.iloc[-1], games=p, all_games=p_all, season=season, projection=proj,
                splits=splits, patterns=patterns, height_corr=corr, opp_ctx=o,
                adjustments=adjust, n_excluded=n_excluded, heat=heat)


def _adjustments(df, absn, p, p_all, opp, out_teammates, out_opp):
    """With/without effects for teammates (this player's own stats) and opposing players
    (how all players performed vs that opponent when the player was out)."""
    tm_f = {s: 1.0 for s in ALL}
    opp_f = {s: 1.0 for s in ALL}
    rows = []
    pid = p_all.player_id.iloc[0]
    for tid in out_teammates:
        out_games = set(absn[(absn.player_id == tid) & (absn.team == p_all.team.iloc[-1])].game_id)
        in_games = set(df[df.player_id == tid].game_id)
        a, b = p[p.game_id.isin(out_games)], p[p.game_id.isin(in_games)]
        name = df.loc[df.player_id == tid, "player"].iloc[-1]
        row = {"kind": "Teammate out", "who": name, "games without": len(a), "games with": len(b)}
        for s in ["min"] + ALL:
            row[s] = _fmt_pair(a[s].mean() if len(a) else np.nan, b[s].mean() if len(b) else np.nan)
        for s in ALL:
            f = _shrunk_ratio(a[s], b[s], len(a), k=6)
            tm_f[s] *= f
        row["applied PTS factor"] = _shrunk_ratio(a["pts"], b["pts"], len(a), 6)
        rows.append(row)
    if len(out_opp):
        q = _ratio_rows(df)
        q = q[q.opp == opp]
        for oid in out_opp:
            out_games = set(absn[(absn.player_id == oid) & (absn.team == opp)].game_id)
            in_games = set(df[df.player_id == oid].game_id)
            a, b = q[q.game_id.isin(out_games)], q[q.game_id.isin(in_games)]
            name = df.loc[df.player_id == oid, "player"].iloc[-1]
            ng = a.game_id.nunique()
            row = {"kind": "Opponent out", "who": name, "games without": ng,
                   "games with": b.game_id.nunique()}
            for s in ALL:
                f = _shrunk_ratio(a[s], b[s], ng, k=4)
                opp_f[s] *= f
                row[s] = _fmt_pair(a[s].mean() if len(a) else np.nan, b[s].mean() if len(b) else np.nan)
            row["applied PTS factor"] = _shrunk_ratio(a["pts"], b["pts"], ng, 4)
            rows.append(row)
    clip = lambda d: {k: float(np.clip(v, 0.75, 1.25)) for k, v in d.items()}  # tiny samples are noisy
    return pd.DataFrame(rows), clip(tm_f), clip(opp_f)


def _heat_factors(h, lean, season):
    """Optional nudge: lean (0-1) x up to +/-10% at heat +/-10, per area."""
    f = {s: 1.0 for s in ALL}
    if not h or lean <= 0:
        return f
    a = h["areas"]
    adj = lambda score: 1 + lean * 0.10 * score / 10
    f["pts"] = adj((a["Scoring"] + a["Shooting efficiency"]) / 2)
    f["reb"], f["ast"] = adj(a["Rebounding"]), adj(a["Playmaking"])
    f["fg3m"] = adj(a["3-point shooting"])
    f["tov"] = 2 - adj(a["Ball security"])
    tot = season["pts"] + season["reb"] + season["ast"]
    f["pra"] = (season["pts"] * f["pts"] + season["reb"] * f["reb"] + season["ast"] * f["ast"]) / tot if tot else 1.0
    return f


def _fmt_pair(without, with_):
    if np.isnan(without) or np.isnan(with_):
        return "n/a"
    return f"{without:.1f} vs {with_:.1f}"


def _context_notes(p_all, p, opp, n_excluded):
    out = []
    if n_excluded:
        kinds = p_all[p_all.abnormal].flag.value_counts()
        out.append(f"Excluded {n_excluded} distorted game(s) from averages and splits "
                   f"({', '.join(f'{v} {k.lower()}' for k, v in kinds.items())}).")
    vs = p_all[p_all.opp == opp]
    if len(vs):
        bad = vs[vs.abnormal | (vs.tm_out != "")]
        for r in bad.itertuples():
            bits = []
            if r.flag:
                bits.append(f"{r.flag.lower()} ({r.min:.0f} min, {r.pf} PF)")
            if r.tm_out:
                bits.append(f"teammates out: {r.tm_out}")
            out.append(f"Game vs {opp} on {r.date:%b %d}: {r.pts} PTS, {r.reb} REB, {r.ast} AST with "
                       + "; ".join(bits) + ", so it may not reflect a normal matchup.")
        if (vs.abnormal.any()) and len(vs) > vs.abnormal.sum():
            c = vs[~vs.abnormal]
            out.append(f"Vs {opp} excluding distorted games ({len(c)}): {c.pts.mean():.1f} PTS, "
                       f"{c.reb.mean():.1f} REB, {c.ast.mean():.1f} AST "
                       f"(all {len(vs)}: {vs.pts.mean():.1f} / {vs.reb.mean():.1f} / {vs.ast.mean():.1f}).")
    return out


def _patterns(p, season, splits, last5, o, opp, corr):
    out = []
    n = len(p)
    for s in ["pts", "reb", "ast", "fg3m"]:
        if season[s] >= 1 and len(last5) >= 5:
            d = last5[s].mean() / season[s] - 1
            if abs(d) >= 0.15:
                out.append(f"Trending {'up' if d > 0 else 'down'} in {LABELS[s]}: last 5 avg "
                           f"{last5[s].mean():.1f} vs {season[s]:.1f} season ({d:+.0%}).")
    for r in splits[splits["matches this game"]].itertuples():
        if r.games < 3:
            continue
        notes = []
        for s in ["pts", "reb", "ast", "fg3m"]:
            if season[s] >= 1:
                d = getattr(r, s) / season[s] - 1
                if abs(d) >= 0.15:
                    notes.append(f"{LABELS[s]} {getattr(r, s):.1f} ({d:+.0%})")
        if notes:
            out.append(f"{r.split} ({r.games} gms): " + ", ".join(notes) + " vs season avg.")
    for s, c in corr.items():
        if c is not None and abs(c) >= 0.3:
            out.append(f"{LABELS[s]} moves with opposing lineup height (r={c:+.2f}): "
                       f"{'higher' if c > 0 else 'lower'} against taller lineups.")
    if n < 10:
        out.append(f"Small sample: only {n} games this season, treat with caution.")
    return out


def opp_rotation(df, opp, top=8):
    t = df[df.team == opp]
    g = t.groupby(["player_id", "player"]).agg(mpg=("min", "mean"), ppg=("pts", "mean"),
                                               rpg=("reb", "mean"), apg=("ast", "mean"),
                                               height_in=("height_in", "first"),
                                               games=("min", "size")).reset_index()
    return g.sort_values("mpg", ascending=False).head(top)


def fmt_height(inches):
    if pd.isna(inches):
        return "?"
    return f"{int(inches // 12)}'{int(round(inches % 12))}\""
