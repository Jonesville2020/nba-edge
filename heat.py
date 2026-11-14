"""Heat gauge: how far a player's recent stretch sits from his own season baseline,
scaled to -10 (the yips) .. +10 (in the zone). Season data only."""
import numpy as np
import pandas as pd

AREAS = ["Shooting efficiency", "3-point shooting", "Free throws", "Scoring",
         "Rebounding", "Playmaking", "Ball security"]
SHOOTING = AREAS[:3]


def _score(z: float) -> float:
    return float(10 * np.tanh(z / 3.5))


def _rate_z(made_r, att_r, made_b, att_b, full_att):
    if att_b < 20 or att_r == 0:
        return 0.0
    p = np.clip(made_b / att_b, 0.02, 0.98)
    z = (made_r / att_r - p) / np.sqrt(p * (1 - p) / att_r)
    return float(z * min(1.0, att_r / full_att))  # few attempts: don't over-read it


def _mean_z(recent: pd.Series, base: pd.Series, sign=1):
    sd = base.std(ddof=1)
    if len(base) < 5 or not sd or np.isnan(sd):
        return 0.0
    return float(sign * (recent.mean() - base.mean()) / (sd / np.sqrt(len(recent))))


def label(score: float) -> str:
    if score >= 7: return "In the zone"
    if score >= 3: return "Hot"
    if score > -3: return "Neutral"
    if score > -7: return "Cold"
    return "The yips"


def compute(p: pd.DataFrame, window: int = 5):
    """p: one player's games sorted by date (distorted games already removed if desired)."""
    need = {"fga", "fta", "fg3a", "fg3m", "ftm"}
    if len(p) < 8 or not need <= set(p.columns):
        return None
    w = min(window, len(p) // 2)
    recent, base = p.tail(w), p.iloc[:-w]
    tsa = lambda g: g.fga.sum() + 0.44 * g.fta.sum()
    ts_r = recent.pts.sum() / (2 * tsa(recent)) if tsa(recent) else 0
    ts_b = base.pts.sum() / (2 * tsa(base)) if tsa(base) else 0
    ts_z = 0.0
    if tsa(base) >= 30 and tsa(recent) > 0:
        ts_z = float((ts_r - ts_b) / np.sqrt(max(ts_b * (1 - ts_b), 0.05) / tsa(recent))
                     * min(1.0, tsa(recent) / 40))
    z = {
        "Shooting efficiency": ts_z,
        "3-point shooting": _rate_z(recent.fg3m.sum(), recent.fg3a.sum(),
                                    base.fg3m.sum(), base.fg3a.sum(), 15),
        "Free throws": _rate_z(recent.ftm.sum(), recent.fta.sum(),
                               base.ftm.sum(), base.fta.sum(), 12),
        "Scoring": _mean_z(recent.pts, base.pts),
        "Rebounding": _mean_z(recent.reb, base.reb),
        "Playmaking": _mean_z(recent.ast, base.ast),
        "Ball security": _mean_z(recent.tov, base.tov, sign=-1),
    }
    weights = {"Shooting efficiency": 1.0, "Scoring": 1.0, "Rebounding": 0.7, "Playmaking": 0.7,
               "3-point shooting": 0.5, "Free throws": 0.3, "Ball security": 0.4}
    overall_z = sum(weights[k] * z[k] for k in z) / np.sqrt(sum(v ** 2 for v in weights.values()))
    luck_keys = {"Shooting efficiency": 1.0, "3-point shooting": 0.5, "Free throws": 0.3}
    luck_z = sum(luck_keys[k] * z[k] for k in luck_keys) / np.sqrt(sum(v ** 2 for v in luck_keys.values()))

    pts_avg = base.pts.mean()
    above = (p.pts > pts_avg).to_numpy()
    streak, last = 0, above[-1]
    for v in above[::-1]:
        if v != last:
            break
        streak += 1
    return dict(
        window=w, overall=_score(overall_z), shooting_luck=_score(luck_z),
        areas={k: _score(v) for k, v in z.items()},
        streak=streak if last else -streak,
        recent_ts=ts_r, base_ts=ts_b,
        recent_3p=recent.fg3m.sum() / max(1, recent.fg3a.sum()),
        base_3p=base.fg3m.sum() / max(1, base.fg3a.sum()))


def board(df: pd.DataFrame, window: int = 5, min_mpg: float = 20, clean=True) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("player_id"):
        if g["min"].mean() < min_mpg:
            continue
        g = g.sort_values("date")
        if clean and (~g.abnormal).sum() >= 8:
            g = g[~g.abnormal]
        h = compute(g, window)
        if h:
            rows.append(dict(player=g.player.iloc[-1], team=g.team.iloc[-1], heat=h["overall"],
                             shooting_luck=h["shooting_luck"], state=label(h["overall"]),
                             streak=h["streak"]))
    return pd.DataFrame(rows).sort_values("heat", ascending=False).reset_index(drop=True)
