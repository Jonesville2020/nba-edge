import os
import pandas as pd
import streamlit as st

import analysis as an
import data
import heat
import model

st.set_page_config(page_title="NBA Edge", layout="wide")
st.markdown("""<style>
.block-container{padding:1rem .8rem 4rem;max-width:900px}
h1,h2,h3{font-family:'Oswald','Arial Narrow',sans-serif;text-transform:uppercase;letter-spacing:.03em}
h2,h3{color:#f2b632}
[data-testid="stMetric"]{background:#0b1a3a;border:1px solid #1c2f5e;border-radius:10px;padding:10px}
[data-testid="stMetricValue"]{color:#f2b632}
.stTabs [data-baseweb="tab"]{font-weight:700}
[data-testid="stDataFrame"]{border:1px solid #1c2f5e;border-radius:10px}
</style>""", unsafe_allow_html=True)
st.title("NBA Edge")

with st.sidebar:
    demo = st.checkbox("Demo mode (synthetic data)", value=False,
                       help="Use if you're offline or the NBA stats site is blocking requests.")
    key = st.text_input("The Odds API key (optional, for Team bets tab)",
                        value=os.getenv("ODDS_API_KEY", ""), type="password")
    half_life = st.slider("Team rating recency half-life (days)", 15, 120, 45)
    min_edge = st.slider("Min edge to flag (pct pts)", 0.0, 10.0, 3.0) / 100
    bankroll = st.number_input("Bankroll ($)", 100, 1_000_000, 1000)
    if st.button("Refresh data"):
        st.cache_data.clear()


@st.cache_data(ttl=3600, show_spinner="Loading this season's player data…")
def load_players(demo_mode: bool):
    logs, bio, team_adv, player_adv = data.demo_player_data() if demo_mode else data.load_player_data()
    df = an.prep(logs, bio)
    ctx = an.team_context(df, team_adv)
    return df, ctx, an.opp_factors(df), player_adv, an.get_absences(df)


@st.cache_data(ttl=3600, show_spinner="Loading games and lines…")
def load_bets(demo_mode: bool, api_key: str):
    if demo_mode:
        return data.demo_games()
    return data.load_games(), data.load_odds(api_key)


names = data.team_names()
tab_player, tab_heat, tab_bets, tab_teams = st.tabs(
    ["Player matchup", "Hot & cold board", "Team bets", "Team ratings"])


def heat_bar(score: float) -> str:
    color = "#e8590c" if score >= 0 else "#1c7ed6"
    pos = (score + 10) / 20 * 100
    left, width = (50, pos - 50) if score >= 0 else (pos, 50 - pos)
    return (f"<div style='position:relative;height:16px;background:#2b2f36;border-radius:8px'>"
            f"<div style='position:absolute;left:{left}%;width:{width}%;height:16px;background:{color};"
            f"border-radius:8px'></div><div style='position:absolute;left:50%;width:2px;height:16px;"
            f"background:#fff'></div></div>")

try:
    df, ctx, factors, player_adv, absn = load_players(demo)
except Exception as e:
    st.warning(f"Live player data unavailable, showing DEMO data instead. {e}")
    df, ctx, factors, player_adv, absn = load_players(True)

# ------------------------------------------------------------------ player
with tab_player:
    roster = (df.sort_values("date").groupby("player_id")
              .agg(player=("player", "last"), team=("team", "last"), games=("min", "size")).reset_index())
    roster["label"] = roster.player + " (" + roster.team + ")"
    roster = roster.sort_values("player")

    c1, c2, c3 = st.columns([3, 3, 2])
    pick = c1.selectbox("Player (type to search)", roster.label.tolist(), index=None,
                        placeholder="Start typing a name…")
    opp_choices = sorted(ctx.index)
    opp = c2.selectbox("Opponent", opp_choices, index=None, placeholder="Choose a team",
                       format_func=lambda a: f"{names.get(a, a)} ({a})")
    home = c3.radio("Location", ["Home", "Away"], horizontal=True) == "Home"

    clean = st.checkbox("Exclude games distorted by foul trouble, blowouts or early exits", value=True)
    h1, h2 = st.columns(2)
    window = h1.slider("Heat window (recent games)", 3, 10, 5)
    lean = h2.slider("Lean into hot/cold streaks in projection", 0, 100, 0,
                     help="0% treats streaks as noise. 100% shifts projections up to 10% toward the streak.") / 100

    if not pick:
        st.info("Pick a player and an opponent. Everything below uses games from this season only.")
    else:
        r = roster[roster.label == pick].iloc[0]
        if opp is None:
            opp = next(t for t in opp_choices if t != r.team)
            st.caption(f"No opponent chosen: showing {opp} as a placeholder.")
        def rotation(team, exclude=None):
            t = an.opp_rotation(df, team, top=12)
            t = t[(t.mpg >= 15) & (t.player_id != exclude)]
            return dict(zip(t.player, t.player_id))

        mates, foes = rotation(r.team, int(r.player_id)), rotation(opp)
        o1, o2 = st.columns(2)
        out_tm = [mates[n] for n in o1.multiselect("Teammates OUT tonight", list(mates))]
        out_opp = [foes[n] for n in o2.multiselect(f"{opp} players OUT tonight", list(foes))]
        res = an.analyze(df, ctx, factors, absn, int(r.player_id), opp, home, out_tm, out_opp, clean, window, lean)
        pl, o = res["player"], res["opp_ctx"]

        st.header(f"{pl.player} vs {names.get(opp, opp)} ({'home' if home else 'away'})")
        pa = player_adv[player_adv.player_id == pl.player_id]
        m = st.columns(6)
        m[0].metric("Games", len(res["games"]))
        m[1].metric("MPG", f"{res['games']['min'].mean():.1f}")
        m[2].metric("Height", an.fmt_height(pl.height_in))
        if not pa.empty:
            a = pa.iloc[0]
            m[3].metric("USG%", f"{a.usg_pct:.1%}")
            m[4].metric("TS%", f"{a.ts_pct:.1%}")
            m[5].metric("PIE", f"{a.pie:.3f}")

        h = res["heat"]
        st.subheader("Heat gauge")
        if h is None:
            st.write("Not enough games or shooting data yet.")
        else:
            g1, g2 = st.columns(2)
            g1.markdown(f"**Overall: {h['overall']:+.1f} ({heat.label(h['overall'])})** over the last "
                        f"{h['window']} games", unsafe_allow_html=False)
            g1.markdown(heat_bar(h["overall"]), unsafe_allow_html=True)
            g2.markdown(f"**Shooting luck: {h['shooting_luck']:+.1f}**: TS% {h['recent_ts']:.1%} vs "
                        f"{h['base_ts']:.1%} before; 3P% {h['recent_3p']:.1%} vs {h['base_3p']:.1%}")
            g2.markdown(heat_bar(h["shooting_luck"]), unsafe_allow_html=True)
            st.caption(f"Streak: {abs(h['streak'])} straight game(s) "
                       f"{'above' if h['streak'] > 0 else 'below'} his usual scoring.")
            cols = st.columns(len(h["areas"]))
            for c, (k, v) in zip(cols, h["areas"].items()):
                c.caption(k)
                c.markdown(f"**{v:+.1f}**")
                c.markdown(heat_bar(v), unsafe_allow_html=True)
            st.caption("Each score compares the recent stretch to the player's own earlier games this "
                       "season, adjusted for how noisy that stat is. 0 is normal, +10 is in the zone, "
                       "-10 is the yips. Shooting luck covers only shooting percentages, which tend to "
                       "drift back toward normal.")

        st.subheader("What to expect")
        proj = res["projection"]
        cards = st.columns(4)
        for col, k in zip(cards, ["pts", "reb", "ast", "pra"]):
            row = proj[proj.key == k].iloc[0]
            col.metric(an.LABELS[k], f"{row.projection:.1f}",
                       f"{row.projection - row['season avg']:+.1f} vs season avg")
            col.caption(f"Typical range {row['low (25%)']:.1f}–{row['high (75%)']:.1f}")
        show = proj.drop(columns="key").set_index("stat")
        st.dataframe(show.style.format("{:.2f}"), width="stretch")
        st.caption("Projection = recency-weighted form × how this opponent has treated players "
                   "relative to their own averages × home/away effect. Range is the middle 50% "
                   "of this player's games. Opp factor above 1 means a favorable matchup.")

        if len(res["adjustments"]):
            st.subheader("Lineup adjustments applied")
            st.dataframe(res["adjustments"].rename(columns=an.LABELS).rename(columns={"min": "MIN"}),
                         hide_index=True, width="stretch")
            st.caption("Cells show the average without vs with that player (this season, clean games). "
                       "Factors are shrunk toward 1 when only a few games are available.")

        st.subheader("Patterns and trends")
        if res["patterns"]:
            for p_ in res["patterns"]:
                st.markdown(f"- {p_}")
        else:
            st.write("No strong deviations found for this matchup.")

        left, right = st.columns(2)
        stat = right.selectbox("Trend stat", an.ALL, format_func=an.LABELS.get)
        with left:
            st.subheader(f"Opponent profile: {opp}")
            if o is not None:
                st.write(f"**{o.def_tier}** (DRtg {o.def_rating:.1f}), **{o.off_tier}** (ORtg {o.off_rating:.1f}), "
                         f"**{o.pace_tier}** ({o.pace:.1f}), **{o.height_tier}** "
                         f"(rotation avg {an.fmt_height(o.height)})")
            rot = an.opp_rotation(df, opp)
            rot["height"] = rot.height_in.map(an.fmt_height)
            st.dataframe(rot[["player", "height", "mpg", "ppg", "rpg", "apg", "games"]]
                         .style.format({"mpg": "{:.1f}", "ppg": "{:.1f}", "rpg": "{:.1f}", "apg": "{:.1f}"}),
                         hide_index=True, width="stretch")
        with right:
            g = res["games"].set_index("date")[[stat]]
            g["5-game avg"] = g[stat].rolling(5, min_periods=1).mean()
            st.line_chart(g)

        st.subheader("Splits")
        sp = res["splits"].rename(columns=an.LABELS)
        numeric = [an.LABELS[s] for s in an.ALL]
        styled = (sp.style.format({c: "{:.1f}" for c in numeric})
                  .apply(lambda row: ["background-color: #1f3a5f" if row["matches this game"] else ""] * len(row),
                         axis=1))
        st.dataframe(styled, hide_index=True, width="stretch")
        st.caption("Highlighted rows apply to this matchup. Small game counts are noisy.")

        hc = res["height_corr"]
        if hc:
            st.subheader("Opposing lineup height vs production")
            st.write(", ".join(f"{an.LABELS[k]}: r={v:+.2f}" for k, v in hc.items() if v is not None)
                     + " (positive = more production against taller lineups)")
            hs = res["games"].dropna(subset=["opp_lineup_h"])
            st.scatter_chart(hs, x="opp_lineup_h", y=stat, x_label="Opposing lineup avg height (in)")

        with st.expander("Game log (with foul trouble / lineup context)"):
            gl = res["all_games"].sort_values("date", ascending=False)
            gl = gl[["date", "opp", "home", "min", "pf", *an.ALL, "flag", "tm_out", "opp_out"]]
            gl = gl.rename(columns={"tm_out": "teammates out", "opp_out": "opp players out"})
            st.dataframe(gl.rename(columns=an.LABELS), hide_index=True, width="stretch")

# -------------------------------------------------------------------- heat
with tab_heat:
    hb = heat.board(df, window=5, clean=clean)
    st.caption("Players averaging 20+ minutes, last 5 games vs their own earlier games this season.")
    cfmt = {"heat": "{:+.1f}", "shooting_luck": "{:+.1f}"}
    hc1, hc2 = st.columns(2)
    hc1.subheader("In the zone")
    hc1.dataframe(hb.head(15).style.format(cfmt), hide_index=True, width="stretch")
    hc2.subheader("The yips")
    hc2.dataframe(hb.tail(15).iloc[::-1].style.format(cfmt), hide_index=True, width="stretch")

# -------------------------------------------------------------------- bets
with tab_bets:
    if not demo and not key:
        st.info("Enter an Odds API key in the sidebar (free at the-odds-api.com) or turn on demo mode.")
    else:
        try:
            games, slate = load_bets(demo, key)
            ratings, hca = model.fit_ratings(games, half_life_days=half_life)
            picks = model.build_picks(slate, ratings, hca)
        except Exception as e:
            st.error(f"Bets data failed to load: {e}")
            picks = pd.DataFrame()
        if picks.empty:
            st.info("No games with lines right now.")
        else:
            show = picks.assign(stake=(picks.kelly * bankroll).round(2))
            cols = ["game", "market", "pick", "line", "price", "model_prob", "market_prob",
                    "edge", "ev", "stake"]
            fmt = {"model_prob": "{:.1%}", "market_prob": "{:.1%}", "edge": "{:+.1%}",
                   "ev": "{:+.1%}", "price": "{:+.0f}", "stake": "${:.2f}"}
            st.subheader("Look at (positive edge)")
            st.dataframe(show[show.edge >= min_edge][cols].style.format(fmt), width="stretch", hide_index=True)
            st.subheader("Avoid (market much better than model)")
            st.dataframe(show[show.edge <= -min_edge].sort_values("edge")[cols[:-1]].style.format(fmt),
                         width="stretch", hide_index=True)
            st.caption("Edge = model win prob minus vig-free market prob. EV per $1. Stake = quarter-Kelly. "
                       "The model ignores injuries and rest, so big edges often mean missing information.")

# ------------------------------------------------------------------- teams
with tab_teams:
    st.dataframe(ctx.reset_index().style.format(
        {"off_rating": "{:.1f}", "def_rating": "{:.1f}", "net_rating": "{:+.1f}", "pace": "{:.1f}",
         "height": "{:.1f}"}), hide_index=True, width="stretch")
