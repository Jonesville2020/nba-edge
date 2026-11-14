"""Ratings model and betting math. Pure functions, no network."""
import math
import numpy as np
import pandas as pd

MARGIN_SD = 12.0  # std dev of NBA final margin around the predicted spread


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fit_ratings(games: pd.DataFrame, half_life_days: float = 45, ridge: float = 5.0):
    """games: columns date, home, away, margin (home - away).
    Returns (ratings dict, home_court_advantage) via recency-weighted ridge regression."""
    teams = sorted(set(games.home) | set(games.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(games)
    X = np.zeros((n, len(teams) + 1))
    for r, (h, a) in enumerate(zip(games.home, games.away)):
        X[r, idx[h]] = 1
        X[r, idx[a]] = -1
    X[:, -1] = 1  # home court intercept
    age = (games.date.max() - games.date).dt.days.to_numpy()
    w = 0.5 ** (age / half_life_days)
    y = games.margin.to_numpy(dtype=float)
    P = np.eye(len(teams) + 1) * ridge
    P[-1, -1] = 0
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + P, XtW @ y)
    ratings = dict(zip(teams, beta[:-1]))
    return ratings, float(beta[-1])


def predict_margin(ratings, hca, home, away) -> float:
    return ratings[home] - ratings[away] + hca


def win_prob(margin: float) -> float:
    return norm_cdf(margin / MARGIN_SD)


def cover_prob(margin: float, home_spread: float) -> float:
    """P(home covers). home_spread is the book line for home (e.g. -5.5)."""
    return norm_cdf((margin + home_spread) / MARGIN_SD)


def implied_prob(american: float) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def devig(p1: float, p2: float):
    s = p1 + p2
    return p1 / s, p2 / s


def decimal_odds(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def ev_per_unit(prob: float, american: float) -> float:
    return prob * (decimal_odds(american) - 1) - (1 - prob)


def kelly(prob: float, american: float, fraction: float = 0.25) -> float:
    b = decimal_odds(american) - 1
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * fraction)


def build_picks(slate: pd.DataFrame, ratings, hca) -> pd.DataFrame:
    """slate: home, away, home_ml, away_ml, home_spread, spread_price (optional).
    Returns one row per bettable side with model prob, market prob, edge, EV."""
    rows = []
    for g in slate.itertuples():
        if g.home not in ratings or g.away not in ratings:
            continue
        m = predict_margin(ratings, hca, g.home, g.away)
        ph = win_prob(m)
        mh, ma = devig(implied_prob(g.home_ml), implied_prob(g.away_ml))
        for side, team, model_p, mkt_p, price in (
            ("Moneyline", g.home, ph, mh, g.home_ml),
            ("Moneyline", g.away, 1 - ph, ma, g.away_ml),
        ):
            rows.append(dict(game=f"{g.away} @ {g.home}", market=side, pick=team,
                             line="ML", price=price, model_prob=model_p,
                             market_prob=mkt_p))
        if not pd.isna(g.home_spread):
            sp = getattr(g, "spread_price", -110)
            sp = -110 if pd.isna(sp) else sp
            ch = cover_prob(m, g.home_spread)
            for team, line, p in ((g.home, g.home_spread, ch), (g.away, -g.home_spread, 1 - ch)):
                rows.append(dict(game=f"{g.away} @ {g.home}", market="Spread", pick=team,
                                 line=f"{line:+g}", price=sp, model_prob=p,
                                 market_prob=0.5))  # two-way -110/-110 devigs to 50%
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["edge"] = df.model_prob - df.market_prob
    df["ev"] = [ev_per_unit(p, o) for p, o in zip(df.model_prob, df.price)]
    df["kelly"] = [kelly(p, o) for p, o in zip(df.model_prob, df.price)]
    return df.sort_values("ev", ascending=False).reset_index(drop=True)
