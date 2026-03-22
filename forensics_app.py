import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from mplsoccer import Pitch
import plotly.express as px
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull
from io import BytesIO
from urllib.parse import quote, urlparse
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION (write only once) ---
_CONFIG_PATH = os.path.join(".streamlit", "config.toml")
if not os.path.exists(_CONFIG_PATH):
    os.makedirs(".streamlit", exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        f.write("""
[theme]
base="dark"
primaryColor="#ff4b4b"
backgroundColor="#0e1117"
secondaryBackgroundColor="#262730"
textColor="#fafafa"
font="sans serif"
[server]
headless = true
""")

# --- 2. SETUP ---
st.set_page_config(page_title="Forensics xG", layout="wide", page_icon="🧬")

# --- 3. DATA LOADER ---
GITHUB_BASE = "https://raw.githubusercontent.com/jczpineda/Forensics-xG/main/"
_ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com"})
_REQUEST_TIMEOUT = 30  # seconds


def _build_url(path_or_url):
    """Build and validate a URL, restricting to allowed hosts."""
    if path_or_url.startswith("http"):
        url = path_or_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    else:
        safe_path = quote(path_or_url)
        url = GITHUB_BASE + safe_path

    parsed = urlparse(url)
    if parsed.hostname not in _ALLOWED_HOSTS:
        return None, f"Untrusted host: {parsed.hostname}"
    return url, None


@st.cache_data(show_spinner=False)
def load_stats_data(path_or_url):
    url, err = _build_url(path_or_url)
    if err:
        return None, err

    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"

        if url.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(r.content))
        else:
            df = pd.read_excel(BytesIO(r.content), engine='openpyxl')

        df.columns = df.columns.astype(str).str.strip()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df, None
    except Exception as e:
        return None, str(e)


# --- Expected Threat (xT) Grid ---
_XT_GRID = np.array([
    [0.00638, 0.00780, 0.00845, 0.00928, 0.01059, 0.01215, 0.01385, 0.01612, 0.01870, 0.02402, 0.02953, 0.03464],
    [0.00679, 0.00843, 0.00923, 0.01019, 0.01166, 0.01349, 0.01548, 0.01816, 0.02120, 0.02756, 0.03508, 0.04097],
    [0.00690, 0.00864, 0.00949, 0.01053, 0.01211, 0.01402, 0.01618, 0.01916, 0.02276, 0.03059, 0.04230, 0.05370],
    [0.00692, 0.00867, 0.00951, 0.01058, 0.01220, 0.01414, 0.01634, 0.01939, 0.02325, 0.03213, 0.05060, 0.25596],
    [0.00692, 0.00867, 0.00951, 0.01058, 0.01220, 0.01414, 0.01634, 0.01939, 0.02325, 0.03213, 0.05060, 0.25596],
    [0.00690, 0.00864, 0.00949, 0.01053, 0.01211, 0.01402, 0.01618, 0.01916, 0.02276, 0.03059, 0.04230, 0.05370],
    [0.00679, 0.00843, 0.00923, 0.01019, 0.01166, 0.01349, 0.01548, 0.01816, 0.02120, 0.02756, 0.03508, 0.04097],
    [0.00638, 0.00780, 0.00845, 0.00928, 0.01059, 0.01215, 0.01385, 0.01612, 0.01870, 0.02402, 0.02953, 0.03464]
])


def _get_xt_vectorized(x_series, y_series):
    """Vectorized xT lookup — much faster than row-by-row apply."""
    x_clipped = np.clip(x_series.values.astype(float), 0.0, 99.9)
    y_clipped = np.clip(y_series.values.astype(float), 0.0, 99.9)
    x_idx = (x_clipped / 100 * 12).astype(int)
    y_idx = (y_clipped / 100 * 8).astype(int)
    return _XT_GRID[y_idx, x_idx]


def _qualifier_ids(qualifiers):
    """Extract qualifier IDs as a set of ints for fast membership testing."""
    ids = set()
    for q in qualifiers:
        try:
            ids.add(int(q['qualifierId']))
        except (KeyError, ValueError, TypeError):
            pass
    return ids


@st.cache_data(show_spinner=False)
def load_match_data(path_or_url):
    url, err = _build_url(path_or_url)
    if err:
        return None, err

    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"

        text = r.content.decode('utf-8', errors='ignore')
        s, e = text.find('('), text.rfind(')')
        if s != -1 and e != -1:
            json_data = json.loads(text[s + 1:e])
        else:
            s, e = text.find('{'), text.rfind('}')
            if s != -1 and e != -1:
                json_data = json.loads(text[s:e + 1])
            else:
                return None, "Invalid JSON"

        events = []
        try:
            contestants = json_data['matchInfo']['contestant']
            home_name, home_id = contestants[0]['name'], contestants[0]['id']
            away_name = contestants[1]['name']
            raw = json_data.get('liveData', {}).get('event', [])

            for i, ev in enumerate(raw):
                try:
                    tid = int(ev.get('typeId', 0))
                    qualifiers = ev.get('qualifier', [])
                    qids = _qualifier_ids(qualifiers)

                    x_start = float(ev.get('x', 0))
                    y_start = float(ev.get('y', 0))
                    end_x = next((float(q['value']) for q in qualifiers if int(q.get('qualifierId', -1)) == 140), x_start)
                    end_y = next((float(q['value']) for q in qualifiers if int(q.get('qualifierId', -1)) == 141), y_start)

                    outcome_val = ev.get('outcome')
                    outcome = "Successful" if outcome_val == 1 else "Unsuccessful"
                    if tid == 16 and 28 in qids:
                        outcome = "Own Goal"

                    team_name = home_name if ev.get('contestantId') == home_id else away_name

                    events.append({
                        "Index": i,
                        "Type": tid,
                        "Player": ev.get('playerName', 'Unknown'),
                        "Team": team_name,
                        "Period": ev.get('periodId'),
                        "x": x_start,
                        "y": y_start,
                        "endX": end_x,
                        "endY": end_y,
                        "Outcome": outcome,
                        "Minute": int(ev.get('timeMin', 0)),
                        "isCross": 2 in qids,
                        "isCorner": 6 in qids,
                        "isFreeKick": 5 in qids,
                        "isFkShot": 26 in qids,
                        "isFastBreak": bool(qids & {23, 24}),
                        "isBlocked": 82 in qids,
                        "isHome": (team_name == home_name)
                    })
                except (KeyError, ValueError, TypeError):
                    continue
        except (KeyError, IndexError):
            return None, "Parse Error"

        df_events = pd.DataFrame(events)

        if not df_events.empty:
            df_events['xT_start'] = _get_xt_vectorized(df_events['x'], df_events['y'])
            df_events['xT_end'] = _get_xt_vectorized(df_events['endX'], df_events['endY'])

            is_pass_or_carry = df_events['Type'].isin([1, 3])
            is_successful = df_events['Outcome'] == 'Successful'
            xt_diff = df_events['xT_end'] - df_events['xT_start']
            df_events['xT_Added'] = np.where(
                is_pass_or_carry & is_successful & (xt_diff > 0),
                xt_diff,
                0
            )

        return df_events, None
    except Exception as e:
        return None, str(e)


# --- 4. DATA INDEX ---
CASE_DATABASE = {
    "RUBEN AMORIM": {
        "json_files": {
            "vs Arsenal": "amorim.json/Arsenal.JSON",
            "vs Aston Villa": "amorim.json/Aston Villa.JSON",
            "vs Bournemouth": "amorim.json/Bournemouth.JSON",
            "vs Brentford": "amorim.json/Brentford.JSON",
            "vs Brighton": "amorim.json/Brighton.JSON",
            "vs Burnley": "amorim.json/Burnley.JSON",
            "vs Chelsea": "amorim.json/Chelsea.JSON",
            "vs Crystal Palace": "amorim.json/Crystal Palace.JSON",
            "vs Everton": "amorim.json/Everton.JSON",
            "vs Fulham": "amorim.json/Fulham.JSON",
            "vs Leeds United": "amorim.json/Leeds United.JSON",
            "vs Liverpool": "amorim.json/Liverpool.JSON",
            "vs Manchester City": "amorim.json/Manchester City.JSON",
            "vs Newcastle United": "amorim.json/Newcastle United.JSON",
            "vs Nottingham Forest": "amorim.json/Nottingham.JSON",
            "vs Sunderland": "amorim.json/Sunderland.JSON",
            "vs Tottenham": "amorim.json/Tottenham.JSON",
            "vs West Ham": "amorim.json/West Ham.JSON",
            "vs Wolverhampton": "amorim.json/Wolverhampton.JSON",
            "vs Wolverhampton (2)": "amorim.json/Wolverhampton 2.JSON"
        },
        "stats_files": {
            "🧤 GK Advanced (Against)": "amorim.csv/Advanced Goalkeeper Stats Against (2025-2026).xlsx",
            "🧤 GK Advanced (For)": "amorim.csv/Advanced Goalkeeper Stats For (2025-2026).xlsx",
            "🧤 GK Standard (Against)": "amorim.csv/Goalkeeper Stats Against (2025-2026).xlsx",
            "🧤 GK Standard (For)": "amorim.csv/Goalkeeper Stats For (2025-2026).xlsx",
            "🎯 Shooting (Against)": "amorim.csv/Shooting Against (2025-2026).xlsx",
            "🎯 Shooting (For)": "amorim.csv/Shooting For (2025-2026).xlsx",
            "⚡ Goal/Shot Creation (Against)": "amorim.csv/Squad Goal and Shot Creation Against (2025-2026).xlsx",
            "⚡ Goal/Shot Creation (For)": "amorim.csv/Squad Goal and Shot Creation For (2025-2026).xlsx",
            "⚽ Passing (Against)": "amorim.csv/Passing Against (2025-2026).xlsx",
            "⚽ Passing (For)": "amorim.csv/Passing For (2025-2026).xlsx",
            "🧠 Passing Types (Against)": "amorim.csv/Passing Types Against (2025-2026).xlsx",
            "🧠 Passing Types (For)": "amorim.csv/Passing Types For (2025-2026).xlsx",
            "⏳ Possession (Against)": "amorim.csv/Possession Against (2025-2026).xlsx",
            "⏳ Possession (For)": "amorim.csv/Possession For (2025-2026).xlsx",
            "🛡️ Squad Defense (Against)": "amorim.csv/Squad Defense Against (2025-2026).xlsx",
            "🛡️ Squad Defense (For)": "amorim.csv/Squad Defense For (2025-2026).xlsx",
            "🏆 Overall Results": "amorim.csv/Overall Results (2025-2026).xlsx",
            "🏠 Home/Away Results": "amorim.csv/Home-Away Results (2025-2026).xlsx",
            "⏱️ Playing Time (Against)": "amorim.csv/Playing Time Against (2025-2026).xlsx",
            "⏱️ Playing Time (For)": "amorim.csv/Playing Time For (2025-2026).xlsx",
            "📊 Standard Stats (Against)": "amorim.csv/Standard Stats Against (2025-2026).xlsx",
            "📊 Standard Stats (For)": "amorim.csv/Standard Stats For (2025-2026).xlsx",
            "🧩 Misc Stats (Against)": "amorim.csv/Miscellaneous Stats Against (2025-2026).xlsx",
            "🧩 Misc Stats (For)": "amorim.csv/Miscellaneous Stats For (2025-2026).xlsx",
        }
    },
    "DARREN FLETCHER": {
        "json_files": {"vs Burnley (2)": "fletcher.json/Burnley 2.JSON"},
        "stats_files": {}
    },
    "MICHAEL CARRICK": {
        "json_files": {
            "vs Arsenal (2)": "carrick.json/Arsenal 2.JSON",
            "vs Aston Villa (2)": "carrick.json/Aston Villa 2.JSON",
            "vs Manchester City (2)": "carrick.json/Manchester City 2.JSON",
            "vs Fulham (2)": "carrick.json/Fulham 2.JSON",
            "vs Tottenham (2)": "carrick.json/Tottenham 2.JSON",
            "vs West Ham (2)": "carrick.json/West Ham 2.JSON",
            "vs Everton (2)": "carrick.json/Everton 2.JSON",
            "vs Newcastle United (2)": "carrick.json/Newcastle United 2.JSON",
            "vs Crystal Palace (2)": "carrick.json/Crystal Palace 2.JSON",
            "vs AFC Bournemouth (2)": "carrick.json/AFC Bournemouth 2.JSON"
        },
        "stats_files": {}
    }
}


@st.cache_data(show_spinner=False)
def _load_all_manager_data(manager_key):
    """Load all matches for a manager. Returns (utd_df, sp_goals, sp_assists)."""
    json_files = CASE_DATABASE[manager_key]["json_files"]
    utd_parts = []
    full_dfs = {}  # label -> full match df
    for label, path in json_files.items():
        df, err = load_match_data(path)
        if df is not None and not df.empty:
            utd = df[df['Team'] == 'Manchester United'].copy()
            if utd.empty:
                continue
            utd['Match'] = label
            utd_parts.append(utd)
            full_dfs[label] = df
    if not utd_parts:
        return pd.DataFrame(), {}, {}
    combined = pd.concat(utd_parts, ignore_index=True)

    # Compute set-piece goals & assists inside the cached function
    sp_goals = {}
    sp_assists = {}
    utd_team = 'Manchester United'
    for match_label, full_df in full_dfs.items():
        goals = full_df[
            (full_df['Type'] == 16) & (full_df['Team'] == utd_team) &
            (full_df['Outcome'] != 'Own Goal') & (full_df['Period'] < 5)
        ]
        sp_events = full_df[
            (full_df['Team'] == utd_team) & (full_df['Type'] == 1) &
            (full_df['isCorner'] | full_df['isFreeKick'])
        ]
        for _, g in goals.iterrows():
            prior_sp = sp_events[
                (sp_events['Index'] < g['Index']) &
                (sp_events['Index'] >= g['Index'] - 15)
            ]
            if prior_sp.empty:
                continue
            sp_evt = prior_sp.iloc[-1]
            between = full_df[
                (full_df['Index'] > sp_evt['Index']) &
                (full_df['Index'] < g['Index'])
            ]
            cleared = between[
                (between['Team'] != utd_team) &
                (between['Type'].isin([10, 11, 52]))
            ]
            if not cleared.empty:
                continue
            scorer = g['Player']
            sp_goals[scorer] = sp_goals.get(scorer, 0) + 1
            pre_goal = full_df[
                (full_df['Index'] < g['Index']) &
                (full_df['Index'] >= g['Index'] - 5) &
                (full_df['Team'] == utd_team) &
                (full_df['Type'] == 1) &
                (full_df['Outcome'] == 'Successful')
            ]
            if not pre_goal.empty:
                assister = pre_goal.iloc[-1]['Player']
                if assister != scorer:
                    sp_assists[assister] = sp_assists.get(assister, 0) + 1
    return combined, sp_goals, sp_assists


# --- 5. HELPER: Zonal Grid Renderer ---
def _draw_zonal_grid(pitch, ax, events_df, x_bins, y_bins, cmap, label_prefix=""):
    """Reusable zonal grid renderer (eliminates duplicate code for defensive/passing grids)."""
    grid_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    max_val = 1
    rows = len(y_bins) - 1
    cols = len(x_bins) - 1

    for i in range(rows):
        for j in range(cols):
            count = len(events_df[
                (events_df['x'] >= x_bins[j]) & (events_df['x'] <= x_bins[j + 1]) &
                (events_df['y'] >= y_bins[i]) & (events_df['y'] <= y_bins[i + 1])
            ])
            grid_data[i, j] = count
            if count > max_val:
                max_val = count

    for i in range(rows):
        for j in range(cols):
            count = int(grid_data[i, j])
            alpha = min(count / max_val, 0.9) if max_val > 0 else 0
            facecolor = cmap(alpha) if count > 0 else '#0e1117'
            width = x_bins[j + 1] - x_bins[j]
            height = y_bins[i + 1] - y_bins[i]
            rect = mpatches.Rectangle(
                (x_bins[j], y_bins[i]), width, height,
                linewidth=1, edgecolor='white', linestyle='--',
                facecolor=facecolor, alpha=0.7, zorder=1
            )
            ax.add_patch(rect)
            if count > 0:
                bbox_props = dict(boxstyle="round,pad=0.3", fc="#333333", ec="none", alpha=0.7)
                ax.text(
                    x_bins[j] + (width / 2), y_bins[i] + (height / 2),
                    str(count), color='white', ha='center', va='center',
                    fontweight='bold', fontsize=18, bbox=bbox_props, zorder=2
                )


def _check_sp_goal(sp_df, team, match_df):
    """Check if a set piece led to a goal (shared helper)."""
    goal_sp = []
    reg_sp = []
    goals = match_df[
        (match_df['Type'] == 16) & (match_df['Team'] == team) &
        (match_df['Outcome'] != 'Own Goal') & (match_df['Period'] < 5)
    ]
    goal_indices = set()

    for _, g in goals.iterrows():
        prior = sp_df[(sp_df['Index'] < g['Index']) & (sp_df['Index'] >= g['Index'] - 15)]
        if not prior.empty:
            sp_idx = prior.iloc[-1]['Index']
            # Check for possession-breaking events between the set piece and goal:
            # goalkeeper save/claim (Type 10/11) or goal kick (Type 52) by the
            # opposing team indicates the set piece was cleared.
            between = match_df[
                (match_df['Index'] > sp_idx) & (match_df['Index'] < g['Index'])
            ]
            cleared = between[
                (between['Team'] != team) &
                (between['Type'].isin([10, 11, 52]))
            ]
            if cleared.empty:
                goal_indices.add(sp_idx)

    for _, p in sp_df.iterrows():
        if p['Index'] in goal_indices:
            goal_sp.append(p)
        else:
            reg_sp.append(p)
    return pd.DataFrame(goal_sp), pd.DataFrame(reg_sp)


# --- 6. INTERFACE ---
st.markdown("""
    <style>
    .tagline { font-size: 24px !important; font-weight: 700 !important; color: #a3a8b8 !important; margin-top: -20px !important; margin-bottom: 30px !important; }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #ff4b4b !important; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #a3a8b8 !important; }
    </style>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 8])
with c1:
    st.markdown("# 🧬")
with c2:
    st.title("FORENSICS xG | CRIME SCENE INVESTIGATION")
    st.markdown('<div class="tagline">Where the Beautiful Game Meets Hard Evidence</div>', unsafe_allow_html=True)
st.divider()

st.subheader("SELECT SUSPECT")

managers = list(CASE_DATABASE.keys())
tabs = st.tabs(managers)

for mgr_idx, manager in enumerate(managers):
    with tabs[mgr_idx]:
        sub_t1, sub_t2, sub_t3 = st.tabs(["📊 STATISTICAL REPORTS", "⚽ MATCH TELEMETRY", "👥 MANCHESTER UNITED PLAYERS"])

        # === TAB A: STATS ===
        with sub_t1:
            stats_files = CASE_DATABASE[manager]["stats_files"]
            if stats_files:
                options = list(stats_files.keys())
                selected_option = st.selectbox("Select Evidence File", options, key=f"s_{manager}")
                target_path = stats_files.get(selected_option, "")

                if target_path:
                    df, err = load_stats_data(target_path)
                    if df is not None:
                        has_90s = '90s' in df.columns
                        c_toggle, _ = st.columns([2, 4])
                        use_per_90 = c_toggle.toggle("⚖️ Normalize Per 90", key=f"p90_{manager}") if has_90s else False

                        display_df = df.copy()
                        if use_per_90:
                            skip_cols = {'90s', 'Year', 'Age', 'Season'}
                            for col in display_df.select_dtypes(include=np.number).columns:
                                if col not in skip_cols:
                                    display_df[col] = (display_df[col] / display_df['90s']).round(2)

                        all_cols = display_df.columns.tolist()
                        num_cols = display_df.select_dtypes(include=np.number).columns.tolist()

                        if all_cols:
                            c1, c2, c3 = st.columns(3)
                            x_col = c1.selectbox("X Axis", all_cols, index=0, key=f"sx_{manager}")
                            y_col = c2.selectbox("Y Axis", num_cols if num_cols else all_cols, index=min(1, len(num_cols) - 1) if len(num_cols) > 1 else 0, key=f"sy_{manager}")
                            lbl_col = c3.selectbox("Label", all_cols, index=0, key=f"sl_{manager}")

                            if y_col in num_cols:
                                fig = px.scatter(display_df, x=x_col, y=y_col, text=lbl_col, template="plotly_dark")
                                fig.update_traces(marker=dict(size=12, color='#ff4b4b', line=dict(width=1, color='white')), textposition='top center')
                                fig.update_layout(height=500, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117')
                                st.plotly_chart(fig, use_container_width=True)

                            st.divider()
                            st.dataframe(display_df, use_container_width=True)
                    elif err:
                        st.error(f"Failed to load data: {err}")
            else:
                st.info("No statistical reports available for this suspect. Please switch to the **MATCH TELEMETRY** tab.")

        # === TAB B: TELEMETRY ===
        with sub_t2:
            json_files = CASE_DATABASE[manager]["json_files"]
            match_options = list(json_files.keys())

            if match_options:
                selected_match = st.selectbox("Select Match", match_options, key=f"m_{manager}")
                target_path = json_files.get(selected_match, "")

                if target_path:
                    with st.spinner("Analyzing Match Data..."):
                        match_df, err = load_match_data(target_path)

                    if match_df is not None and not match_df.empty:
                        home_team_rows = match_df[match_df['isHome']]
                        away_team_rows = match_df[~match_df['isHome']]
                        home_name = home_team_rows['Team'].iloc[0] if not home_team_rows.empty else "Unknown Home"
                        away_name = away_team_rows['Team'].iloc[0] if not away_team_rows.empty else "Unknown Away"

                        st.header(f"{home_name} vs {away_name}")

                        teams = match_df['Team'].unique()
                        c1, c2, c3 = st.columns(3)
                        sel_team = c1.selectbox("Squad", teams, key=f"st_{manager}")

                        team_players = sorted([
                            p for p in match_df[match_df['Team'] == sel_team]['Player'].unique()
                            if str(p) != 'Unknown'
                        ])
                        team_players.insert(0, "All Players")
                        sel_player = c2.selectbox("Player", team_players, key=f"sp_{manager}")

                        max_minute = int(match_df['Minute'].max()) if not match_df.empty else 95
                        min_range = c3.slider("Minute Range", 0, max_minute, (0, max_minute), key=f"sper_{manager}")

                        # --- Prepare filtered DataFrames ---
                        plot_df = match_df[match_df['Team'] == sel_team].copy()
                        plot_df = plot_df[(plot_df['Minute'] >= min_range[0]) & (plot_df['Minute'] <= min_range[1])]

                        if sel_player != "All Players":
                            plot_df = plot_df[plot_df['Player'] == sel_player]

                        opp_team = [t for t in teams if t != sel_team][0]
                        opp_stats = match_df[match_df['Team'] == opp_team].copy()
                        opp_stats = opp_stats[(opp_stats['Minute'] >= min_range[0]) & (opp_stats['Minute'] <= min_range[1])]

                        # Flip opponent coordinates so they attack left (x=0)
                        opp_stats['x'] = 100 - opp_stats['x']
                        opp_stats['y'] = 100 - opp_stats['y']
                        opp_stats['endX'] = 100 - opp_stats['endX']
                        opp_stats['endY'] = 100 - opp_stats['endY']

                        attack_label = "Attacking Direction ➡️"
                        viz_df = plot_df.copy()

                        # --- Metrics ---
                        opp_passes_count = len(opp_stats[opp_stats['Type'] == 1])
                        my_def_actions = len(plot_df[plot_df['Type'].isin([4, 7, 8])])
                        ppda = round(opp_passes_count / my_def_actions, 1) if my_def_actions > 0 else 0

                        team_f3 = len(plot_df[(plot_df['Type'] == 1) & (plot_df['x'] > 66.6)])
                        opp_f3 = len(opp_stats[(opp_stats['Type'] == 1) & (opp_stats['x'] < 33.3)])
                        total_f3 = team_f3 + opp_f3
                        field_tilt = round((team_f3 / total_f3 * 100), 1) if total_f3 > 0 else 0

                        total_passes = len(plot_df[plot_df['Type'] == 1])
                        succ_passes = len(plot_df[(plot_df['Type'] == 1) & (plot_df['Outcome'] == 'Successful')])
                        pass_acc = round((succ_passes / total_passes * 100), 1) if total_passes > 0 else 0
                        total_shots = len(plot_df[plot_df['Type'].isin([13, 14, 15, 16])])
                        goals = len(plot_df[plot_df['Type'] == 16])
                        tackles = len(plot_df[plot_df['Type'] == 7])
                        total_def = tackles + len(plot_df[plot_df['Type'] == 8]) + len(plot_df[plot_df['Type'] == 12])
                        fouls = len(plot_df[plot_df['Type'] == 4])

                        recoveries = plot_df[
                            ((plot_df['Type'] == 7) & (plot_df['Outcome'] == 'Successful')) |
                            (plot_df['Type'].isin([8, 12]))
                        ]
                        avg_rec_height = round(recoveries['x'].mean(), 1) if not recoveries.empty else 0

                        st.divider()
                        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                        m1.metric("⚽ Passing", f"{succ_passes}/{total_passes}", f"{pass_acc}%")
                        m2.metric("🎯 Shooting", f"{goals} Goals", f"{total_shots} Shots")
                        m3.metric("🛡️ Def. Actions", f"{total_def}", f"{tackles} Tackles")
                        m4.metric("⚠️ Discipline", f"{fouls} Fouls", "Committed")
                        m5.metric("⚖️ Field Tilt", f"{field_tilt}%", "Final 3rd Share")
                        m6.metric("🛑 PPDA", f"{ppda}", "Passes per Def. Action")
                        m7.metric("📏 Def. Line", f"{avg_rec_height}m", "Avg Recovery Height")
                        st.divider()

                        # --- Module Selection ---
                        st.markdown("#### 🗂️ Select Evidence Layers")

                        off_options = [
                            "Actions Leading to Shots", "Counter Attacks (Rapid Transitions)", "Creator Map (Shot Assists)",
                            "Attacking Zones (5 Lanes)", "Zone Invasions", "The Pulse (Shot Race)", "The Breakout (Progressive Carries)",
                            "The Air Raid (Crossing Zones)", "Average Offensive Positions", "Matchup: Offense vs Opp. Defense",
                            "High-Value Actions (xT Map)", "Expected Threat (xT) Grid", "Cumulative Threat Race (xT)",
                            "Player Impact Board (xT Created)", "Transition Passing Breakdown"
                        ]
                        def_options = [
                            "Defensive Shield (Heatmap + Line)", "Duel Map (Tackles Won/Lost)", "Impact Zone (Convex Hull)",
                            "Defensive Actions", "Average Defensive Positions", "Matchup: Defense vs Opp. Offense",
                            "Zonal Defensive Pressure Map", "Defensive Penetration Conceded", "Zones of Responsibility (Voronoi)"
                        ]
                        pos_options = [
                            "The Architect (Build-Up Phase)", "Momentum Map", "Game Control (Possession)", "Zone 14 & Half-Spaces",
                            "Passing Network (Structure)", "Pass Sonar (Radar)", "Pass Map", "Passing Heatmap", "General Heatmap",
                            "Zonal Passing Control Map", "Switches of Play (Long Diagonals)"
                        ]
                        set_options = [
                            "Set Piece Targeting (Corners)", "Free Kick Targeting"
                        ]

                        c_off, c_def, c_pos, c_set = st.columns(4)
                        with c_off:
                            off_mods = st.multiselect("⚔️ Offensive", off_options, default=["Expected Threat (xT) Grid"], key=f"mo_{manager}")
                        with c_def:
                            def_mods = st.multiselect("🛡️ Defensive", def_options, key=f"md_{manager}")
                        with c_pos:
                            pos_mods = st.multiselect("⚽ Possession", pos_options, key=f"mp_{manager}")
                        with c_set:
                            set_mods = st.multiselect("🎯 Set-Pieces", set_options, key=f"ms_{manager}")

                        modules = off_mods + def_mods + pos_mods + set_mods

                        # --- NON-PITCH MODULES (CHARTS) ---
                        if "Pass Sonar (Radar)" in modules:
                            st.subheader("📡 Pass Sonar (Distribution Angles)")
                            if sel_player == "All Players":
                                st.info("⚠️ Please select a specific player from the dropdown above to view their Pass Sonar.")
                            else:
                                p_passes = plot_df[(plot_df['Type'] == 1) & (plot_df['Outcome'] == 'Successful')]
                                if not p_passes.empty:
                                    dx = p_passes['endX'] - p_passes['x']
                                    dy = p_passes['endY'] - p_passes['y']
                                    angles = np.arctan2(dy, dx)
                                    fig_polar = plt.figure(figsize=(5, 5))
                                    fig_polar.set_facecolor('#0e1117')
                                    ax_polar = fig_polar.add_subplot(111, polar=True)
                                    ax_polar.set_facecolor('#0e1117')
                                    ax_polar.hist(angles, bins=24, color='#00ffff', alpha=0.7, edgecolor='white')
                                    ax_polar.set_theta_zero_location('E')
                                    ax_polar.set_yticks([])
                                    ax_polar.grid(color='#262730')
                                    ax_polar.tick_params(axis='x', colors='white')
                                    st.pyplot(fig_polar)
                                    plt.close(fig_polar)
                                else:
                                    st.info("Not enough successful passes for sonar distribution in this timeframe.")

                        if "Cumulative Threat Race (xT)" in modules:
                            st.subheader("📈 Cumulative Threat Race (xT)")
                            xt_race = match_df[match_df['Type'].isin([1, 3])].sort_values("Index").copy()
                            xt_race = xt_race[(xt_race['Minute'] >= min_range[0]) & (xt_race['Minute'] <= min_range[1])]
                            xt_race['xT_Pos'] = np.where(xt_race['xT_Added'] > 0, xt_race['xT_Added'], 0)
                            xt_race['Cumulative xT'] = xt_race.groupby('Team')['xT_Pos'].cumsum()

                            fig_xt = px.line(
                                xt_race, x='Minute', y='Cumulative xT', color='Team',
                                title='Expected Threat (xT) Accumulation', line_shape='hv',
                                color_discrete_map={sel_team: '#ff4b4b'}, template="plotly_dark"
                            )
                            fig_xt.update_traces(selector=dict(name=opp_team), line_color='grey')
                            st.plotly_chart(fig_xt, use_container_width=True)

                        if "Player Impact Board (xT Created)" in modules:
                            st.subheader(f"📊 Player Impact Board: {sel_team}")
                            player_xt = plot_df[plot_df['Type'].isin([1, 3])].groupby('Player')['xT_Added'].sum().reset_index()
                            player_xt = player_xt.sort_values('xT_Added', ascending=True)

                            fig_bar = px.bar(
                                player_xt, x='xT_Added', y='Player', orientation='h',
                                title='Total Threat Created (Positive xT Added)', template="plotly_dark",
                                color='xT_Added', color_continuous_scale=["#ff4b4b", "#00ff85"]
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        if "Game Control (Possession)" in modules:
                            st.subheader("🎮 Game Control: Possession % (Rolling 5-min)")
                            full = match_df.copy()
                            full = full[(full['Minute'] >= min_range[0]) & (full['Minute'] <= min_range[1])]

                            if not full.empty:
                                full['MinBin'] = (full['Minute'] // 5) * 5
                                poss_data = full[full['Type'] == 1].groupby(['MinBin', 'Team']).size().reset_index(name='Passes')
                                poss_pivot = poss_data.pivot(index='MinBin', columns='Team', values='Passes').fillna(0)
                                poss_pivot['Total'] = poss_pivot.sum(axis=1)
                                if sel_team in poss_pivot.columns:
                                    poss_pivot['Possession'] = (poss_pivot[sel_team] / poss_pivot['Total']) * 100
                                    fig_gc = px.area(
                                        poss_pivot, x=poss_pivot.index, y='Possession',
                                        labels={'MinBin': 'Minute', 'Possession': f'{sel_team} Possession %'},
                                        template="plotly_dark", color_discrete_sequence=['#ff4b4b']
                                    )
                                    fig_gc.update_yaxes(range=[0, 100])
                                    fig_gc.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5)
                                    st.plotly_chart(fig_gc, use_container_width=True)

                        if "Momentum Map" in modules:
                            st.subheader("📈 Momentum Gap: Home vs Away")
                            mom_base = match_df.sort_values("Index").copy()
                            mom_base = mom_base[(mom_base['Minute'] >= min_range[0]) & (mom_base['Minute'] <= min_range[1])]

                            if not mom_base.empty:
                                mom_base['FinalThirdEntry'] = np.where(
                                    (mom_base['Type'] == 1) & (mom_base['Outcome'] == 'Successful') & (mom_base['endX'] > 66.6),
                                    1, 0
                                ).astype(int)
                                mom_base['CumulativeThreat'] = mom_base.groupby('Team')['FinalThirdEntry'].cumsum()
                                fig_mom = px.line(
                                    mom_base, x='Index', y='CumulativeThreat', color='Team',
                                    labels={'Index': 'Match Progression', 'CumulativeThreat': 'Attacking Pressure'},
                                    color_discrete_map={sel_team: '#ff4b4b'}, template="plotly_dark"
                                )
                                fig_mom.update_traces(selector=dict(name=opp_team), line_color='grey')
                                fig_mom.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', legend_title_text='')
                                st.plotly_chart(fig_mom, use_container_width=True)

                        if "The Pulse (Shot Race)" in modules:
                            st.subheader("📈 The Pulse: Shot Accumulation Race")
                            shot_race = match_df[match_df['Type'].isin([13, 14, 15, 16])].sort_values("Index").copy()
                            shot_race = shot_race[(shot_race['Minute'] >= min_range[0]) & (shot_race['Minute'] <= min_range[1])]

                            if not shot_race.empty:
                                shot_race['Count'] = 1
                                shot_race['Cumulative Shots'] = shot_race.groupby('Team')['Count'].cumsum()
                                fig_pulse = px.line(
                                    shot_race, x='Minute', y='Cumulative Shots', color='Team',
                                    title='Shot Race', labels={'Minute': 'Match Minute', 'Cumulative Shots': 'Total Shots Taken'},
                                    line_shape='hv', color_discrete_map={sel_team: '#ff4b4b'}, template="plotly_dark"
                                )
                                fig_pulse.update_traces(selector=dict(name=opp_team), line_color='grey')
                                fig_pulse.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                                st.plotly_chart(fig_pulse, use_container_width=True)
                            else:
                                st.write("No shots recorded in this interval.")

                        # --- PITCH MODULES ---
                        non_pitch_modules = {
                            "Game Control (Possession)", "Momentum Map", "The Pulse (Shot Race)",
                            "Cumulative Threat Race (xT)", "Player Impact Board (xT Created)", "Pass Sonar (Radar)"
                        }
                        pitch_modules = [m for m in modules if m not in non_pitch_modules]

                        if pitch_modules:
                            st.markdown(f"<h3 style='text-align: center; color: white;'>{attack_label}</h3>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 7))
                            fig.set_facecolor('#0e1117')
                            ax.set_facecolor('#0e1117')
                            pitch = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                            pitch.draw(ax=ax)

                            # --- Set Pieces ---
                            if "Free Kick Targeting" in pitch_modules:
                                fks = viz_df[(viz_df['Type'] == 1) & (viz_df['isFreeKick'])]
                                fk_shots = viz_df[(viz_df['Type'].isin([13, 14, 15, 16])) & (viz_df['isFkShot'])]

                                if not fks.empty or not fk_shots.empty:
                                    if not fks.empty:
                                        goal_fks, reg_fks = _check_sp_goal(fks, sel_team, match_df)

                                        if not reg_fks.empty:
                                            succ_fk = reg_fks[reg_fks['Outcome'] == 'Successful']
                                            fail_fk = reg_fks[reg_fks['Outcome'] == 'Unsuccessful']

                                            if not succ_fk.empty:
                                                pitch.arrows(succ_fk['x'].values, succ_fk['y'].values, succ_fk['endX'].values, succ_fk['endY'].values,
                                                             width=2, headwidth=4, color='#00ffff', alpha=0.8, ax=ax, label=f'Succ. FK Pass ({len(succ_fk)})')
                                                pitch.scatter(succ_fk['x'].values, succ_fk['y'].values, s=40, color='#00ffff', edgecolors='white', ax=ax)

                                            if not fail_fk.empty:
                                                pitch.arrows(fail_fk['x'].values, fail_fk['y'].values, fail_fk['endX'].values, fail_fk['endY'].values,
                                                             width=2, headwidth=4, color='#ff4b4b', alpha=0.5, ax=ax, label=f'Failed FK Pass ({len(fail_fk)})')
                                                pitch.scatter(fail_fk['x'].values, fail_fk['y'].values, s=40, color='#ff4b4b', edgecolors='white', ax=ax)

                                        if not goal_fks.empty:
                                            pitch.arrows(goal_fks['x'].values, goal_fks['y'].values, goal_fks['endX'].values, goal_fks['endY'].values,
                                                         width=3, headwidth=5, color='#00ff85', alpha=1.0, ax=ax, label=f'Goal-Creating FK ({len(goal_fks)})', zorder=5)
                                            pitch.scatter(goal_fks['endX'].values, goal_fks['endY'].values, s=250, marker='*', color='#00ff85', edgecolors='black', ax=ax, zorder=6)

                                    if not fk_shots.empty:
                                        fk_goals = fk_shots[(fk_shots['Type'] == 16) & (fk_shots['Outcome'] != 'Own Goal')]
                                        fk_misses = fk_shots[~((fk_shots['Type'] == 16) & (fk_shots['Outcome'] != 'Own Goal'))]

                                        if not fk_misses.empty:
                                            pitch.scatter(fk_misses['x'].values, fk_misses['y'].values, s=200, marker='X', color='#ffd700', edgecolors='black', ax=ax, label=f'Direct FK Miss ({len(fk_misses)})', zorder=4)
                                        if not fk_goals.empty:
                                            pitch.scatter(fk_goals['x'].values, fk_goals['y'].values, s=400, marker='*', color='#00ff85', edgecolors='black', ax=ax, label=f'Direct FK Goal ({len(fk_goals)})', zorder=6)

                                    ax.legend(facecolor='#262730', labelcolor='white', loc='upper right', bbox_to_anchor=(1, 1.15), ncol=2, fontsize=8)
                                else:
                                    pitch.annotate("No Free Kicks recorded in this range", xy=(50, 50), c='white', ha='center', va='center', size=12, ax=ax)

                            if "Set Piece Targeting (Corners)" in pitch_modules:
                                corners = viz_df[(viz_df['Type'] == 1) & (viz_df['isCorner'])]
                                if not corners.empty:
                                    goal_crn, reg_crn = _check_sp_goal(corners, sel_team, match_df)

                                    if not reg_crn.empty:
                                        succ_crn = reg_crn[reg_crn['Outcome'] == 'Successful']
                                        fail_crn = reg_crn[reg_crn['Outcome'] == 'Unsuccessful']

                                        if not succ_crn.empty:
                                            pitch.arrows(succ_crn['x'].values, succ_crn['y'].values, succ_crn['endX'].values, succ_crn['endY'].values,
                                                         width=2, headwidth=4, color='#00ffff', alpha=0.6, ax=ax, label=f'Succ. Corner ({len(succ_crn)})')
                                        if not fail_crn.empty:
                                            pitch.arrows(fail_crn['x'].values, fail_crn['y'].values, fail_crn['endX'].values, fail_crn['endY'].values,
                                                         width=2, headwidth=4, color='#ff4b4b', alpha=0.5, ax=ax, label=f'Failed Corner ({len(fail_crn)})')

                                    if not goal_crn.empty:
                                        pitch.arrows(goal_crn['x'].values, goal_crn['y'].values, goal_crn['endX'].values, goal_crn['endY'].values,
                                                     width=3, headwidth=5, color='#00ff85', alpha=1.0, ax=ax, label=f'Goal-Creating Corner ({len(goal_crn)})', zorder=5)
                                        pitch.scatter(goal_crn['endX'].values, goal_crn['endY'].values, s=250, marker='*', color='#00ff85', edgecolors='black', ax=ax, zorder=6)

                                    ax.legend(facecolor='#262730', labelcolor='white', loc='upper right', bbox_to_anchor=(1, 1.15), ncol=2, fontsize=8)
                                else:
                                    pitch.annotate("No Corners recorded in this range", xy=(50, 50), c='white', ha='center', va='center', size=12, ax=ax)

                            # --- xT Modules ---
                            if "High-Value Actions (xT Map)" in pitch_modules:
                                high_xt = viz_df[(viz_df['Type'].isin([1, 3])) & (viz_df['Outcome'] == 'Successful') & (viz_df['xT_Added'] > 0.015)]

                                if not high_xt.empty:
                                    high_passes = high_xt[high_xt['Type'] == 1]
                                    if not high_passes.empty:
                                        pitch.arrows(high_passes['x'].values, high_passes['y'].values, high_passes['endX'].values, high_passes['endY'].values,
                                                     width=3, headwidth=5, color='#00ff85', alpha=0.9, ax=ax, label='High Value Pass (>0.015 xT)')
                                        pitch.scatter(high_passes['x'].values, high_passes['y'].values, s=50, color='#00ff85', edgecolors='white', ax=ax)

                                    high_dribbles = high_xt[high_xt['Type'] == 3]
                                    if not high_dribbles.empty:
                                        pitch.lines(high_dribbles['x'].values, high_dribbles['y'].values, high_dribbles['endX'].values, high_dribbles['endY'].values,
                                                    lw=3, linestyle='dashed', color='#00ffff', alpha=0.9, ax=ax, label='High Value Carry')
                                        pitch.scatter(high_dribbles['x'].values, high_dribbles['y'].values, s=80, marker='d', color='#00ffff', edgecolors='white', ax=ax)

                                    ax.legend(facecolor='#262730', labelcolor='white', loc='upper left')
                                else:
                                    pitch.annotate("No actions exceeded +0.015 xT in this range", xy=(50, 50), c='white', ha='center', va='center', size=12, ax=ax)

                            if "Expected Threat (xT) Grid" in pitch_modules:
                                xt_acts = viz_df[(viz_df['Type'].isin([1, 3])) & (viz_df['Outcome'] == 'Successful') & (viz_df['xT_Added'] > 0)]

                                if not xt_acts.empty:
                                    bin_statistic = pitch.bin_statistic(xt_acts['x'].values, xt_acts['y'].values, values=xt_acts['xT_Added'].values, statistic='sum', bins=(12, 8))
                                    pitch.heatmap(bin_statistic, ax=ax, cmap='magma', alpha=0.7, edgecolors='#262730', lw=1, zorder=0)

                                    stat = bin_statistic['statistic']
                                    cx = bin_statistic['cx']
                                    cy = bin_statistic['cy']

                                    for row_i in range(stat.shape[0]):
                                        for col_j in range(stat.shape[1]):
                                            val = stat[row_i, col_j]
                                            if val > 0.001:
                                                ax.text(cx[row_i, col_j], cy[row_i, col_j], f'{val:.3f}', color='white',
                                                        ha='center', va='center', fontsize=9, fontweight='bold', zorder=1)
                                else:
                                    pitch.annotate("No positive xT actions recorded", xy=(50, 50), c='white', ha='center', va='center', size=15, ax=ax)

                            # --- Offensive Modules ---
                            if "Actions Leading to Shots" in pitch_modules:
                                sca_filter = st.selectbox(
                                    "Filter Shot Outcome:",
                                    ["All Attempts", "Goals Only", "Saved Only", "Blocked Only", "Missed Only"],
                                    key=f"sca_filter_{manager}"
                                )
                                shots_df = viz_df[viz_df['Type'].isin([13, 14, 15, 16])]

                                if sca_filter == "Goals Only":
                                    shots_df = shots_df[(shots_df['Type'] == 16) & (shots_df['Outcome'] != 'Own Goal')]
                                elif sca_filter == "Saved Only":
                                    shots_df = shots_df[(shots_df['Type'] == 15) & (~shots_df['isBlocked'])]
                                elif sca_filter == "Blocked Only":
                                    shots_df = shots_df[shots_df['isBlocked']]
                                elif sca_filter == "Missed Only":
                                    shots_df = shots_df[~((shots_df['Type'] == 16) & (shots_df['Outcome'] != 'Own Goal')) & (~shots_df['isBlocked']) & (shots_df['Type'] != 15)]

                                for _, shot in shots_df.iterrows():
                                    shot_idx = shot['Index']
                                    shot_type = shot['Type']

                                    if shot_type == 16 and shot['Outcome'] != 'Own Goal':
                                        color, marker, size = '#00ff85', '*', 400
                                    elif shot['isBlocked']:
                                        color, marker, size = '#aaaaaa', 'X', 200
                                    elif shot_type == 15:
                                        color, marker, size = '#ffd700', 'o', 200
                                    else:
                                        color, marker, size = '#ff4b4b', 'x', 200

                                    pitch.scatter(shot.x, shot.y, s=size, marker=marker, c=color, edgecolors='white', ax=ax, zorder=4)

                                    recent_events = viz_df[(viz_df['Index'] < shot_idx) & (viz_df['Index'] >= shot_idx - 10)]
                                    recent_passes = recent_events[(recent_events['Type'] == 1) & (recent_events['Outcome'] == 'Successful')]

                                    if not recent_passes.empty:
                                        last_pass = recent_passes.iloc[-1]
                                        pitch.arrows(last_pass.x, last_pass.y, shot.x, shot.y,
                                                     width=2, headwidth=5, color=color, alpha=0.5, ax=ax, zorder=3)

                                legend_elements = [
                                    mlines.Line2D([0], [0], marker='*', color='w', label='Goal', markerfacecolor='#00ff85', markersize=15, linestyle='None'),
                                    mlines.Line2D([0], [0], marker='o', color='w', label='Saved', markerfacecolor='#ffd700', markersize=10, linestyle='None'),
                                    mlines.Line2D([0], [0], marker='X', color='w', label='Blocked', markerfacecolor='#aaaaaa', markersize=10, linestyle='None'),
                                    mlines.Line2D([0], [0], marker='x', color='w', label='Missed', markerfacecolor='#ff4b4b', markersize=10, linestyle='None'),
                                    mlines.Line2D([0], [0], color='w', lw=2, alpha=0.5, label='Creation Vector')
                                ]
                                ax.legend(handles=legend_elements, facecolor='#262730', labelcolor='white', loc='upper left')

                            if "Counter Attacks (Rapid Transitions)" in pitch_modules:
                                fb_indices = plot_df[(plot_df['Type'].isin([13, 14, 15, 16])) & (plot_df['isFastBreak'])].index
                                fb_viz = viz_df.loc[fb_indices]

                                fb_goals = pd.DataFrame()
                                fb_shots = pd.DataFrame()
                                if not fb_viz.empty:
                                    fb_goals = fb_viz[(fb_viz['Type'] == 16) & (fb_viz['Outcome'] != 'Own Goal')]
                                    fb_shots = fb_viz[~((fb_viz['Type'] == 16) & (fb_viz['Outcome'] != 'Own Goal'))]

                                buildup_passes = pd.DataFrame()
                                if not fb_viz.empty:
                                    blist = []
                                    for _, row in fb_viz.iterrows():
                                        ev_idx = row['Index']
                                        seq = match_df[
                                            (match_df['Index'] >= ev_idx - 50) & (match_df['Index'] < ev_idx) &
                                            (match_df['Team'] == sel_team) & (match_df['Type'] == 1) &
                                            (match_df['Outcome'] == 'Successful')
                                        ]
                                        blist.append(seq)
                                    if blist:
                                        buildup_passes = pd.concat(blist).drop_duplicates()

                                trans_indices = plot_df[
                                    (plot_df['Type'] == 1) & (plot_df['Outcome'] == 'Successful') &
                                    (plot_df['x'] < 40) & (plot_df['endX'] > 70)
                                ].index
                                trans_viz = viz_df.loc[trans_indices]

                                if not trans_viz.empty:
                                    pitch.arrows(trans_viz['x'].values, trans_viz['y'].values, trans_viz['endX'].values, trans_viz['endY'].values,
                                                 width=2, color='#ffd700', alpha=0.5, ax=ax, label=f'Direct Counter Pass: {len(trans_viz)}')
                                if not buildup_passes.empty:
                                    pitch.arrows(buildup_passes['x'].values, buildup_passes['y'].values, buildup_passes['endX'].values, buildup_passes['endY'].values,
                                                 width=3, color='#00ffff', alpha=0.8, ax=ax, label=f'Fast Break Sequence: {len(buildup_passes)}')
                                if not fb_shots.empty:
                                    pitch.scatter(fb_shots['x'].values, fb_shots['y'].values, s=400, marker='*', c='#00ffff', edgecolors='white', ax=ax, label=f'Fast Break Shot: {len(fb_shots)}')
                                if not fb_goals.empty:
                                    pitch.scatter(fb_goals['x'].values, fb_goals['y'].values, s=600, marker='*', c='#00ff85', edgecolors='white', linewidth=1.5, ax=ax, label=f'Fast Break Goal: {len(fb_goals)}')
                                if not trans_viz.empty or not fb_viz.empty or not buildup_passes.empty:
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Creator Map (Shot Assists)" in pitch_modules:
                                shots_for_assists = viz_df[viz_df['Type'].isin([13, 14, 15, 16])]
                                assists = []
                                key_passes = []

                                for _, shot in shots_for_assists.iterrows():
                                    shot_idx = shot['Index']
                                    prev_events = viz_df[(viz_df['Index'] < shot_idx) & (viz_df['Index'] >= shot_idx - 5)]
                                    prev_passes = prev_events[(prev_events['Type'] == 1) & (prev_events['Outcome'] == 'Successful')]

                                    if not prev_passes.empty:
                                        key_pass = prev_passes.iloc[-1]
                                        if shot['Type'] == 16 and shot['Outcome'] != 'Own Goal':
                                            assists.append(key_pass)
                                        else:
                                            key_passes.append(key_pass)

                                if key_passes:
                                    kp_df = pd.DataFrame(key_passes)
                                    pitch.arrows(kp_df['x'].values, kp_df['y'].values, kp_df['endX'].values, kp_df['endY'].values,
                                                 width=2, headwidth=4, color='#00ffff', alpha=0.7, ax=ax, label=f'Key Pass ({len(kp_df)})')
                                    pitch.scatter(kp_df['x'].values, kp_df['y'].values, s=60, color='#00ffff', edgecolors='white', ax=ax, zorder=3)

                                if assists:
                                    ast_df = pd.DataFrame(assists)
                                    pitch.arrows(ast_df['x'].values, ast_df['y'].values, ast_df['endX'].values, ast_df['endY'].values,
                                                 width=3, headwidth=5, color='#ffd700', alpha=0.9, ax=ax, label=f'Assist ({len(ast_df)})')
                                    pitch.scatter(ast_df['x'].values, ast_df['y'].values, s=200, marker='*', color='#ffd700', edgecolors='black', ax=ax, zorder=4)

                                if key_passes or assists:
                                    ax.legend(facecolor='#262730', labelcolor='white')
                                else:
                                    pitch.annotate("No Shot Assists Recorded", xy=(50, 50), c='white', ha='center', va='center', size=15, ax=ax)

                            if "Attacking Zones (5 Lanes)" in pitch_modules:
                                f3_acts = plot_df[(plot_df['x'] > 66.6) & (plot_df['Type'] == 1) & (plot_df['Outcome'] == 'Successful')]
                                total_f3_acts = len(f3_acts)
                                if total_f3_acts > 0:
                                    lane_defs = [
                                        ("Right Flank", 0, 20),
                                        ("Right Half", 20, 37),
                                        ("Center", 37, 63),
                                        ("Left Half", 63, 80),
                                        ("Left Flank", 80, 100),
                                    ]
                                    for label, y_lo, y_hi in lane_defs:
                                        count = len(f3_acts[(f3_acts['y'] > y_lo) & (f3_acts['y'] <= y_hi)]) if y_lo > 0 else len(f3_acts[f3_acts['y'] <= y_hi])
                                        pct = (count / total_f3_acts) * 100
                                        height = y_hi - y_lo
                                        rect = mpatches.Rectangle((66.6, y_lo), 33.4, height, alpha=min(0.9, max(0.2, pct / 40)), color='#ff4b4b', ec='white')
                                        ax.add_patch(rect)
                                        ax.text(66.6 + 16.7, y_lo + (height / 2), f"{label}\n{pct:.1f}%", color='white', ha='center', va='center', fontweight='bold', fontsize=9)

                            if "Zone Invasions" in pitch_modules:
                                zi_passes = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful')]
                                zi_dribbles_won = viz_df[(viz_df['Type'] == 3) & (viz_df['Outcome'] == 'Successful')]

                                f3_pass = zi_passes[(zi_passes['endX'] > 66) & (zi_passes['endX'] <= 83)]
                                box_pass = zi_passes[zi_passes['endX'] > 83]
                                drib_won = zi_dribbles_won[zi_dribbles_won['x'] > 66]

                                if not f3_pass.empty:
                                    pitch.arrows(f3_pass['x'].values, f3_pass['y'].values, f3_pass['endX'].values, f3_pass['endY'].values, width=3, color='white', alpha=0.6, ax=ax, label='Into F3')
                                if not box_pass.empty:
                                    pitch.arrows(box_pass['x'].values, box_pass['y'].values, box_pass['endX'].values, box_pass['endY'].values, width=3, color='#ffd700', alpha=0.9, ax=ax, label='Into Box')
                                if not drib_won.empty:
                                    pitch.scatter(drib_won['x'].values, drib_won['y'].values, s=150, marker='d', c='#00ffff', edgecolors='white', ax=ax, label='Succ. Dribble')
                                ax.legend(facecolor='#262730', labelcolor='white')

                            if "The Breakout (Progressive Carries)" in pitch_modules:
                                prog_indices = plot_df[plot_df['endX'] > plot_df['x'] + 10].index
                                carries_viz = viz_df.loc[prog_indices]
                                if not carries_viz.empty:
                                    pitch.lines(carries_viz['x'].values, carries_viz['y'].values, carries_viz['endX'].values, carries_viz['endY'].values,
                                                lw=3, linestyle='dashed', color='#00ffff', alpha=0.8, ax=ax, label='Prog. Carry (>10m)')
                                    pitch.scatter(carries_viz['x'].values, carries_viz['y'].values, s=50, c='#00ffff', ax=ax)
                                    ax.legend(facecolor='#262730', labelcolor='white', title=f"Count: {len(carries_viz)}")

                            if "The Air Raid (Crossing Zones)" in pitch_modules:
                                crosses = viz_df[(viz_df['Type'] == 1) & (viz_df['isCross'])]
                                if not crosses.empty:
                                    succ = crosses[crosses['Outcome'] == 'Successful']
                                    fail = crosses[crosses['Outcome'] == 'Unsuccessful']
                                    if not succ.empty:
                                        pitch.arrows(succ['x'].values, succ['y'].values, succ['endX'].values, succ['endY'].values, width=2, color='#00ff85', label='Succ. Cross', ax=ax)
                                    if not fail.empty:
                                        pitch.arrows(fail['x'].values, fail['y'].values, fail['endX'].values, fail['endY'].values, width=2, color='#ff4b4b', alpha=0.5, label='Failed Cross', ax=ax)
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Transition Passing Breakdown" in pitch_modules:
                                trans_passes = viz_df[
                                    (viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful') &
                                    ((viz_df['isFastBreak']) | ((viz_df['x'] < 66.6) & (viz_df['endX'] - viz_df['x'] >= 20)))
                                ]
                                tp_shots = viz_df[viz_df['Type'].isin([13, 14, 15, 16])]
                                tp_goals = tp_shots[(tp_shots['Type'] == 16) & (tp_shots['Outcome'] != 'Own Goal')]
                                tp_misses = tp_shots[~((tp_shots['Type'] == 16) & (tp_shots['Outcome'] != 'Own Goal'))]

                                if not trans_passes.empty:
                                    pitch.arrows(trans_passes['x'].values, trans_passes['y'].values, trans_passes['endX'].values, trans_passes['endY'].values,
                                                 width=2, headwidth=4, color='#ffd700', alpha=0.8, ax=ax, label='Transition Passes', zorder=2)
                                    bbox_props = dict(boxstyle="round,pad=0.5", fc="#1e1e1e", ec="white", lw=1)
                                    ax.text(80, 5, f"Total Transition Passes: {len(trans_passes)}", color='white', ha='center', va='center', fontweight='bold', fontsize=10, bbox=bbox_props, zorder=5)

                                if not tp_misses.empty:
                                    pitch.scatter(tp_misses['x'].values, tp_misses['y'].values, s=150, marker='X', color='#ff4b4b', ax=ax, label='Shot (No Goal)', zorder=4)
                                if not tp_goals.empty:
                                    pitch.scatter(tp_goals['x'].values, tp_goals['y'].values, s=300, marker='*', color='#00ff85', edgecolors='white', ax=ax, label='Goal', zorder=5)
                                ax.legend(facecolor='#262730', labelcolor='white', loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

                            # --- Average Positions ---
                            if "Average Offensive Positions" in pitch_modules:
                                off_events = viz_df[viz_df['Type'].isin([1, 3])]
                                if not off_events.empty:
                                    off_avg = off_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(off_avg['x'].values, off_avg['y'].values, s=150, c='#00ffff', edgecolors='white', ax=ax, zorder=3, label='Avg Offensive Position')
                                    for player, row in off_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Average Defensive Positions" in pitch_modules:
                                def_events = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                if not def_events.empty:
                                    def_avg = def_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(def_avg['x'].values, def_avg['y'].values, s=150, c='#ff4b4b', edgecolors='white', marker='s', ax=ax, zorder=3, label='Avg Defensive Position')
                                    for player, row in def_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            # --- Matchup Modules (define opp_viz once for both) ---
                            needs_opp_viz = ("Matchup: Offense vs Opp. Defense" in pitch_modules or "Matchup: Defense vs Opp. Offense" in pitch_modules)
                            opp_viz = opp_stats.copy() if needs_opp_viz else pd.DataFrame()

                            if "Matchup: Offense vs Opp. Defense" in pitch_modules:
                                off_events = viz_df[viz_df['Type'].isin([1, 3])]
                                opp_def_events = opp_viz[opp_viz['Type'].isin([4, 7, 8, 12])] if not opp_viz.empty else pd.DataFrame()

                                if not off_events.empty:
                                    off_avg = off_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(off_avg['x'].values, off_avg['y'].values, s=150, c='#00ffff', edgecolors='white', ax=ax, zorder=3, label=f'{sel_team} (Off)')
                                    for player, row in off_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)

                                if not opp_def_events.empty:
                                    def_avg = opp_def_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(def_avg['x'].values, def_avg['y'].values, s=150, c='#ff4b4b', edgecolors='white', marker='s', ax=ax, zorder=3, label=f'{opp_team} (Def)')
                                    for player, row in def_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)
                                ax.legend(facecolor='#262730', labelcolor='white')

                            if "Matchup: Defense vs Opp. Offense" in pitch_modules:
                                def_events = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                opp_off_events = opp_viz[opp_viz['Type'].isin([1, 3])] if not opp_viz.empty else pd.DataFrame()

                                if not def_events.empty:
                                    def_avg = def_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(def_avg['x'].values, def_avg['y'].values, s=150, c='#ff4b4b', edgecolors='white', marker='s', ax=ax, zorder=3, label=f'{sel_team} (Def)')
                                    for player, row in def_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)

                                if not opp_off_events.empty:
                                    off_avg = opp_off_events.groupby('Player')[['x', 'y']].mean()
                                    pitch.scatter(off_avg['x'].values, off_avg['y'].values, s=150, c='#00ffff', edgecolors='white', ax=ax, zorder=3, label=f'{opp_team} (Off)')
                                    for player, row in off_avg.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y + 3), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)
                                ax.legend(facecolor='#262730', labelcolor='white')

                            # --- Defensive Modules ---
                            if "Defensive Shield (Heatmap + Line)" in pitch_modules:
                                def_heat = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                if not def_heat.empty:
                                    pitch.kdeplot(def_heat['x'].values, def_heat['y'].values, ax=ax, cmap='viridis', fill=True, levels=100, alpha=0.6)

                                if avg_rec_height > 0:
                                    draw_x = avg_rec_height
                                    pitch.lines(draw_x, 0, draw_x, 100, color='#ffd700', lw=3, linestyle='dashed', alpha=0.9, ax=ax, label=f'Avg Def Line ({avg_rec_height}m)')
                                    ax.add_patch(mpatches.Rectangle((0, 0), draw_x, 100, alpha=0.1, color='#ffd700', ec=None))
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Duel Map (Tackles Won/Lost)" in pitch_modules:
                                duels = viz_df[viz_df['Type'] == 7]
                                if not duels.empty:
                                    won = duels[duels['Outcome'] == 'Successful']
                                    lost = duels[duels['Outcome'] == 'Unsuccessful']
                                    if not won.empty:
                                        pitch.scatter(won['x'].values, won['y'].values, s=200, marker='p', c='#00ff85', edgecolors='black', ax=ax, label='Tackle Won')
                                    if not lost.empty:
                                        pitch.scatter(lost['x'].values, lost['y'].values, s=200, marker='X', c='#ff4b4b', edgecolors='white', ax=ax, label='Tackle Lost')
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Impact Zone (Convex Hull)" in pitch_modules:
                                def_pts = viz_df[viz_df['Type'].isin([4, 7, 8, 12])][['x', 'y']]
                                if len(def_pts) >= 3:
                                    hull = ConvexHull(def_pts[['x', 'y']].values)
                                    hull_pts = def_pts[['x', 'y']].values[hull.vertices]
                                    hull_pts = np.vstack((hull_pts, hull_pts[0]))
                                    poly = mpatches.Polygon(hull_pts, closed=True, facecolor='#ff4b4b', alpha=0.3, edgecolor='white', lw=2)
                                    ax.add_patch(poly)
                                    pitch.scatter(def_pts['x'].values, def_pts['y'].values, s=50, c='#ff4b4b', ax=ax, alpha=0.6)

                            if "Defensive Actions" in pitch_modules:
                                def_act = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                if not def_act.empty:
                                    tackles_df = def_act[def_act['Type'] == 7]
                                    interceptions_df = def_act[def_act['Type'] == 8]
                                    fouls_df = def_act[def_act['Type'] == 4]
                                    if not tackles_df.empty:
                                        pitch.scatter(tackles_df['x'].values, tackles_df['y'].values, s=150, marker='d', c='#3399ff', edgecolors='white', ax=ax, label='Tackle')
                                    if not interceptions_df.empty:
                                        pitch.scatter(interceptions_df['x'].values, interceptions_df['y'].values, s=150, marker='s', c='#ff9900', edgecolors='black', ax=ax, label='Interception')
                                    if not fouls_df.empty:
                                        pitch.scatter(fouls_df['x'].values, fouls_df['y'].values, s=150, marker='X', c='#ff4b4b', ax=ax, label='Foul')
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Zonal Defensive Pressure Map" in pitch_modules:
                                zd_events = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                _draw_zonal_grid(pitch, ax, zd_events,
                                                 x_bins=[0, 33.33, 66.67, 100],
                                                 y_bins=[0, 21.1, 78.9, 100],
                                                 cmap=plt.cm.Reds)

                            if "Zonal Passing Control Map" in pitch_modules:
                                zp_events = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful')]
                                _draw_zonal_grid(pitch, ax, zp_events,
                                                 x_bins=[0, 33.33, 66.67, 100],
                                                 y_bins=[0, 21.1, 78.9, 100],
                                                 cmap=plt.cm.Blues)

                            if "Defensive Penetration Conceded" in pitch_modules:
                                opp_succ_passes = opp_stats[(opp_stats['Type'] == 1) & (opp_stats['Outcome'] == 'Successful')]
                                box_passes = opp_succ_passes[
                                    (opp_succ_passes['endX'] <= 17) & (opp_succ_passes['x'] > 17) &
                                    (opp_succ_passes['endY'] >= 21.1) & (opp_succ_passes['endY'] <= 78.9)
                                ]
                                z14_passes = opp_succ_passes[
                                    (opp_succ_passes['endX'] > 17) & (opp_succ_passes['endX'] <= 35) &
                                    (opp_succ_passes['x'] > 35) & (opp_succ_passes['endY'] >= 37) &
                                    (opp_succ_passes['endY'] <= 63)
                                ]

                                ax.add_patch(mpatches.Rectangle((0, 21.1), 17, 57.8, alpha=0.15, color='#ff4b4b', ec=None, zorder=1))
                                ax.add_patch(mpatches.Rectangle((17, 37), 18, 26, alpha=0.15, color='#ffd700', ec=None, zorder=1))

                                def_acts_pen = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                                if not def_acts_pen.empty:
                                    pitch.scatter(def_acts_pen['x'].values, def_acts_pen['y'].values, s=30, marker='s', color='#262730', edgecolors='white', alpha=0.6, ax=ax, zorder=2, label='Our Def. Actions')

                                def _split_by_goal(pass_df, scoring_team):
                                    fatal, safe = [], []
                                    goal_evts = match_df[
                                        (match_df['Type'] == 16) & (match_df['Team'] == scoring_team) &
                                        (match_df['Outcome'] != 'Own Goal') & (match_df['Period'] < 5)
                                    ]
                                    fatal_indices = set()
                                    for _, g in goal_evts.iterrows():
                                        prior = pass_df[(pass_df['Index'] < g['Index']) & (pass_df['Index'] >= g['Index'] - 15)]
                                        if not prior.empty:
                                            fatal_indices.add(prior.iloc[-1]['Index'])
                                    for _, p in pass_df.iterrows():
                                        (fatal if p['Index'] in fatal_indices else safe).append(p)
                                    return pd.DataFrame(fatal), pd.DataFrame(safe)

                                fatal_box, safe_box = _split_by_goal(box_passes, opp_team)
                                fatal_z14, safe_z14 = _split_by_goal(z14_passes, opp_team)

                                if not safe_box.empty:
                                    pitch.arrows(safe_box['x'].values, safe_box['y'].values, safe_box['endX'].values, safe_box['endY'].values,
                                                 width=2, headwidth=4, color='#ff4b4b', alpha=0.6, ax=ax, label=f'Box Conceded ({len(safe_box)})', zorder=4)
                                    pitch.scatter(safe_box['x'].values, safe_box['y'].values, s=40, color='#ff4b4b', edgecolors='white', ax=ax, zorder=4)
                                if not fatal_box.empty:
                                    pitch.arrows(fatal_box['x'].values, fatal_box['y'].values, fatal_box['endX'].values, fatal_box['endY'].values,
                                                 width=3, headwidth=5, color='#00ff85', alpha=1.0, ax=ax, label=f'Fatal Box Pass ({len(fatal_box)})', zorder=5)
                                    pitch.scatter(fatal_box['endX'].values, fatal_box['endY'].values, s=250, marker='*', color='#00ff85', edgecolors='black', ax=ax, zorder=6)
                                if not safe_z14.empty:
                                    pitch.arrows(safe_z14['x'].values, safe_z14['y'].values, safe_z14['endX'].values, safe_z14['endY'].values,
                                                 width=2, headwidth=4, color='#ffd700', alpha=0.6, ax=ax, label=f'Pocket Conceded ({len(safe_z14)})', zorder=3)
                                    pitch.scatter(safe_z14['x'].values, safe_z14['y'].values, s=40, color='#ffd700', edgecolors='black', ax=ax, zorder=3)
                                if not fatal_z14.empty:
                                    pitch.arrows(fatal_z14['x'].values, fatal_z14['y'].values, fatal_z14['endX'].values, fatal_z14['endY'].values,
                                                 width=3, headwidth=5, color='#00ff85', alpha=1.0, ax=ax, label=f'Fatal Pocket Pass ({len(fatal_z14)})', zorder=5)
                                    pitch.scatter(fatal_z14['endX'].values, fatal_z14['endY'].values, s=250, marker='*', color='#00ff85', edgecolors='black', ax=ax, zorder=6)

                                if box_passes.empty and z14_passes.empty:
                                    pitch.annotate("No Major Defensive Penetrations Conceded", xy=(50, 50), c='white', ha='center', va='center', size=12, ax=ax)
                                else:
                                    ax.legend(facecolor='#262730', labelcolor='white', loc='upper right', bbox_to_anchor=(1, 1.15), ncol=2, fontsize=8)

                            if "Zones of Responsibility (Voronoi)" in pitch_modules:
                                team_avg = viz_df.groupby('Player')[['x', 'y']].mean().reset_index()
                                if len(team_avg) >= 4:
                                    x_vals = team_avg['x'].values
                                    y_vals = team_avg['y'].values
                                    try:
                                        team_vor, _ = pitch.voronoi(x_vals, y_vals, team_avg['Player'].values)
                                        pitch.polygon(team_vor, ax=ax, fc='#ff4b4b', ec='white', lw=2, alpha=0.2, zorder=1)
                                    except (ValueError, TypeError):
                                        team_vor = pitch.voronoi(x_vals, y_vals)
                                        pitch.polygon(team_vor, ax=ax, fc='#ff4b4b', ec='white', lw=2, alpha=0.2, zorder=1)
                                    pitch.scatter(x_vals, y_vals, s=150, c='#ff4b4b', edgecolors='white', ax=ax, zorder=3)
                                    for _, row in team_avg.iterrows():
                                        pitch.annotate(row['Player'].split(" ")[-1], xy=(row.x, row.y + 2.5), c='white', ha='center', va='center', size=9, fontweight='bold', ax=ax, zorder=4)

                            # --- Possession Modules ---
                            if "The Architect (Build-Up Phase)" in pitch_modules:
                                build_up_indices = plot_df[(plot_df['Type'] == 1) & (plot_df['x'] < 33)].index
                                build_up_viz = viz_df.loc[build_up_indices]

                                if not build_up_viz.empty:
                                    norm_endX = plot_df.loc[build_up_indices, 'endX']
                                    circ = build_up_viz[norm_endX < 33]
                                    prog = build_up_viz[(norm_endX >= 33) & (norm_endX < 66)]
                                    launch = build_up_viz[norm_endX >= 66]

                                    if not circ.empty:
                                        pitch.lines(circ['x'].values, circ['y'].values, circ['endX'].values, circ['endY'].values, color='white', alpha=0.1, lw=2, ax=ax, label=f'Circulation: {len(circ)}')
                                    if not prog.empty:
                                        pitch.lines(prog['x'].values, prog['y'].values, prog['endX'].values, prog['endY'].values, color='#00ffff', alpha=0.6, lw=3, ax=ax, label=f'Progression: {len(prog)}')
                                        pitch.scatter(prog['endX'].values, prog['endY'].values, s=30, c='#00ffff', ax=ax)
                                    if not launch.empty:
                                        pitch.lines(launch['x'].values, launch['y'].values, launch['endX'].values, launch['endY'].values, color='#ff00ff', alpha=0.8, lw=3, ax=ax, label=f'Long Ball: {len(launch)}')
                                        pitch.scatter(launch['endX'].values, launch['endY'].values, s=30, c='#ff00ff', ax=ax)
                                    ax.legend(facecolor='#262730', labelcolor='white')

                            if "Zone 14 & Half-Spaces" in pitch_modules:
                                zones = {
                                    "Zone 14": {"x": (65, 85), "y": (37, 63), "color": "#ffd700"},
                                    "LHS": {"x": (65, 85), "y": (20, 37), "color": "#ff4b4b"},
                                    "RHS": {"x": (65, 85), "y": (63, 80), "color": "#ff4b4b"}
                                }
                                legend_handles = []

                                for zone_name, bounds in zones.items():
                                    xmin, xmax = bounds["x"]
                                    ymin, ymax = bounds["y"]
                                    c = bounds["color"]

                                    rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, alpha=0.15, color=c, ec='white', lw=1.5, zorder=1)
                                    ax.add_patch(rect)

                                    zone_events = viz_df[
                                        (viz_df['x'] >= xmin) & (viz_df['x'] <= xmax) &
                                        (viz_df['y'] >= ymin) & (viz_df['y'] <= ymax)
                                    ]

                                    z_succ_passes = zone_events[(zone_events['Type'] == 1) & (zone_events['Outcome'] == 'Successful')]
                                    z_succ_dribbles = zone_events[(zone_events['Type'] == 3) & (zone_events['Outcome'] == 'Successful')]
                                    z_fail_passes = zone_events[(zone_events['Type'] == 1) & (zone_events['Outcome'] == 'Unsuccessful')]
                                    z_fail_dribbles = zone_events[(zone_events['Type'] == 3) & (zone_events['Outcome'] == 'Unsuccessful')]

                                    succ_count = len(z_succ_passes) + len(z_succ_dribbles)
                                    loss_count = len(z_fail_passes) + len(z_fail_dribbles)

                                    if not z_succ_passes.empty:
                                        pitch.arrows(z_succ_passes['x'].values, z_succ_passes['y'].values, z_succ_passes['endX'].values, z_succ_passes['endY'].values,
                                                     width=2, headwidth=4, color='white', alpha=0.8, ax=ax, zorder=3)
                                    if not z_succ_dribbles.empty:
                                        pitch.scatter(z_succ_dribbles['x'].values, z_succ_dribbles['y'].values, s=100, marker='d', color='#00ffff', edgecolors='white', ax=ax, zorder=4)
                                    if not z_fail_passes.empty:
                                        pitch.arrows(z_fail_passes['x'].values, z_fail_passes['y'].values, z_fail_passes['endX'].values, z_fail_passes['endY'].values,
                                                     width=2, headwidth=4, color='#ff4b4b', alpha=0.4, ax=ax, zorder=2)
                                    if not z_fail_dribbles.empty:
                                        pitch.scatter(z_fail_dribbles['x'].values, z_fail_dribbles['y'].values, s=100, marker='x', color='#ff4b4b', ax=ax, zorder=4)

                                    legend_handles.append(mpatches.Patch(color=c, alpha=0.5, label=f"{zone_name} | Succ: {succ_count} | Lost: {loss_count}"))

                                legend_handles.extend([
                                    mlines.Line2D([0], [0], color='white', lw=2, label='Succ Pass'),
                                    mlines.Line2D([0], [0], marker='d', color='w', markerfacecolor='#00ffff', markersize=8, linestyle='None', label='Succ Dribble'),
                                    mlines.Line2D([0], [0], color='#ff4b4b', alpha=0.4, lw=2, label='Failed Pass'),
                                    mlines.Line2D([0], [0], marker='x', color='w', markeredgecolor='#ff4b4b', markersize=8, linestyle='None', label='Failed Dribble'),
                                ])
                                ax.legend(handles=legend_handles, facecolor='#262730', labelcolor='white', loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3, fontsize=8)

                            if "Passing Network (Structure)" in pitch_modules:
                                net_df = viz_df.copy()
                                avg_pos = net_df.groupby('Player')[['x', 'y']].mean()
                                pass_counts = net_df.groupby('Player').size()
                                net_df['NextPlayer'] = net_df['Player'].shift(-1)
                                net_df['NextTeam'] = net_df['Team'].shift(-1)
                                connections = net_df[(net_df['Type'] == 1) & (net_df['Outcome'] == 'Successful') & (net_df['NextTeam'] == sel_team)]
                                if not connections.empty:
                                    edges = connections.groupby(['Player', 'NextPlayer']).size().reset_index(name='count')
                                    edges = edges[edges['count'] > 2]
                                    for _, row in edges.iterrows():
                                        p1, p2 = row['Player'], row['NextPlayer']
                                        if p1 in avg_pos.index and p2 in avg_pos.index:
                                            width = row['count'] * 0.5
                                            pitch.lines(avg_pos.loc[p1].x, avg_pos.loc[p1].y, avg_pos.loc[p2].x, avg_pos.loc[p2].y, lw=width, color='#ff4b4b', alpha=0.6, ax=ax, zorder=1)
                                    pitch.scatter(avg_pos.x, avg_pos.y, s=pass_counts * 5, color='#0e1117', edgecolors='white', linewidth=2, ax=ax, zorder=2)
                                    for player, row in avg_pos.iterrows():
                                        pitch.annotate(player.split(" ")[-1], xy=(row.x, row.y), c='white', va='center', ha='center', size=8, ax=ax, zorder=3)

                            if "Pass Map" in pitch_modules:
                                pm_passes = viz_df[viz_df['Type'] == 1]
                                if not pm_passes.empty:
                                    succ = pm_passes[pm_passes['Outcome'] == 'Successful']
                                    fail = pm_passes[pm_passes['Outcome'] == 'Unsuccessful']
                                    if not succ.empty:
                                        pitch.arrows(succ['x'].values, succ['y'].values, succ['endX'].values, succ['endY'].values, width=2, color='#00ff85', alpha=0.3, ax=ax)
                                    if not fail.empty:
                                        pitch.arrows(fail['x'].values, fail['y'].values, fail['endX'].values, fail['endY'].values, width=2, color='#ff4b4b', alpha=0.3, ax=ax)

                            if "Passing Heatmap" in pitch_modules:
                                passes_heat = viz_df[viz_df['Type'] == 1]
                                if not passes_heat.empty:
                                    pitch.kdeplot(passes_heat['x'].values, passes_heat['y'].values, ax=ax, cmap='plasma', fill=True, levels=100, alpha=0.6)

                            if "General Heatmap" in pitch_modules and not viz_df.empty:
                                pitch.kdeplot(viz_df['x'].values, viz_df['y'].values, ax=ax, cmap='hot', fill=True, levels=100, alpha=0.6)

                            # Switches of Play: >= 50 Opta units ≈ ~34m lateral distance
                            if "Switches of Play (Long Diagonals)" in pitch_modules:
                                sw_passes = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful')]
                                switches = sw_passes[abs(sw_passes['endY'] - sw_passes['y']) >= 50]

                                if not switches.empty:
                                    pitch.arrows(switches['x'].values, switches['y'].values, switches['endX'].values, switches['endY'].values,
                                                 width=2, headwidth=4, color='#00529F', alpha=0.9, ax=ax, zorder=3)
                                    pitch.scatter(switches['x'].values, switches['y'].values, s=30, color='white', edgecolors='black', ax=ax, zorder=4)

                                    bbox_props = dict(boxstyle="round,pad=0.5", fc="#00529F", ec="white", lw=2)
                                    ax.text(85, 5, f"Total Switches: {len(switches)}", color='white', ha='center', va='center', fontweight='bold', fontsize=10, bbox=bbox_props, zorder=5)

                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.error(f"Error: {err}")

        # === TAB C: MANCHESTER UNITED PLAYERS ===
        with sub_t3:
            mgr_short = manager.split()[-1].title()  # Amorim / Fletcher / Carrick
            st.header(f"👥 Manchester United Players")
            with st.spinner(f"Loading all {mgr_short}-era match data..."):
                utd_df, sp_goals_map, sp_assists_map = _load_all_manager_data(manager)

            if not utd_df.empty:
                n_matches = utd_df['Match'].nunique()
                players = sorted([p for p in utd_df['Player'].unique() if p != 'Unknown'])

                # --- Player Stats Summary ---
                st.subheader(f"📊 Average Player Statistics ({n_matches} Matches)")
                stats_rows = []
                for player in players:
                    pdf = utd_df[utd_df['Player'] == player]
                    mp = pdf['Match'].nunique()
                    passes_att = len(pdf[pdf['Type'] == 1])
                    passes_comp = len(pdf[(pdf['Type'] == 1) & (pdf['Outcome'] == 'Successful')])
                    pass_pct = round(passes_comp / passes_att * 100, 1) if passes_att > 0 else 0
                    shots = len(pdf[pdf['Type'].isin([13, 14, 15, 16])])
                    goals = len(pdf[(pdf['Type'] == 16) & (pdf['Outcome'] != 'Own Goal')])
                    tackles = len(pdf[pdf['Type'] == 7])
                    interceptions = len(pdf[pdf['Type'] == 8])
                    clearances = len(pdf[pdf['Type'] == 12])
                    fouls = len(pdf[pdf['Type'] == 4])
                    xt = pdf['xT_Added'].sum()
                    stats_rows.append({
                        'Player': player,
                        'Matches': mp,
                        'Passes/Match': round(passes_att / mp, 1),
                        'Comp/Match': round(passes_comp / mp, 1),
                        'Pass %': pass_pct,
                        'Shots/Match': round(shots / mp, 2),
                        'Goals': goals,
                        'SP Goals': sp_goals_map.get(player, 0),
                        'SP Assists': sp_assists_map.get(player, 0),
                        'Tackles/Match': round(tackles / mp, 2),
                        'Int/Match': round(interceptions / mp, 2),
                        'Clear/Match': round(clearances / mp, 2),
                        'Fouls/Match': round(fouls / mp, 2),
                        'xT/Match': round(xt / mp, 3),
                    })
                stats_summary = pd.DataFrame(stats_rows).sort_values('xT/Match', ascending=False)
                st.dataframe(stats_summary, use_container_width=True, hide_index=True)
                st.divider()

                # --- Player Selector ---
                sel_p = st.selectbox("Select Player for Visualizations", players, key=f"player_sel_{manager}")
                p_df = utd_df[utd_df['Player'] == sel_p]

                if not p_df.empty:
                    col_hm, col_ps = st.columns(2)

                    # --- Average Heatmap ---
                    with col_hm:
                        st.subheader("🔥 Average Heatmap")
                        fig_hm, ax_hm = plt.subplots(figsize=(10, 7))
                        fig_hm.set_facecolor('#0e1117')
                        ax_hm.set_facecolor('#0e1117')
                        pitch_hm = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                        pitch_hm.draw(ax=ax_hm)
                        if len(p_df) >= 2:
                            pitch_hm.kdeplot(p_df['x'].values, p_df['y'].values, ax=ax_hm, cmap='hot', fill=True, levels=100, alpha=0.6)
                        else:
                            pitch_hm.scatter(p_df['x'].values, p_df['y'].values, s=100, color='#ff4b4b', ax=ax_hm)
                        ax_hm.set_title(f'{sel_p} — All {mgr_short} Matches ({p_df["Match"].nunique()} games)', color='white', fontsize=12)
                        st.pyplot(fig_hm)
                        plt.close(fig_hm)

                    # --- Average Pass Sonar ---
                    with col_ps:
                        st.subheader("📡 Average Pass Sonar")
                        p_passes = p_df[(p_df['Type'] == 1) & (p_df['Outcome'] == 'Successful')]
                        if not p_passes.empty:
                            dx = p_passes['endX'] - p_passes['x']
                            dy = p_passes['endY'] - p_passes['y']
                            angles = np.arctan2(dy, dx)
                            fig_ps = plt.figure(figsize=(6, 6))
                            fig_ps.set_facecolor('#0e1117')
                            ax_ps = fig_ps.add_subplot(111, polar=True)
                            ax_ps.set_facecolor('#0e1117')
                            ax_ps.hist(angles, bins=24, color='#00ffff', alpha=0.7, edgecolor='white')
                            ax_ps.set_theta_zero_location('E')
                            ax_ps.set_yticks([])
                            ax_ps.grid(color='#262730')
                            ax_ps.tick_params(axis='x', colors='white')
                            ax_ps.set_title(f'{sel_p} — Pass Direction', color='white', fontsize=12, pad=20)
                            st.pyplot(fig_ps)
                            plt.close(fig_ps)
                        else:
                            st.info("No successful passes recorded for this player.")

                    st.divider()

                    # --- Passing Network (average positions + connections) ---
                    st.subheader("🔗 Average Passing Network")
                    net_df = utd_df.copy()
                    avg_pos = net_df.groupby('Player')[['x', 'y']].mean()
                    pass_counts = net_df.groupby('Player').size()
                    net_df['NextPlayer'] = net_df['Player'].shift(-1)
                    net_df['NextTeam'] = net_df['Team'].shift(-1)
                    connections = net_df[
                        (net_df['Type'] == 1) & (net_df['Outcome'] == 'Successful') &
                        (net_df['NextTeam'] == 'Manchester United')
                    ]
                    min_edge_count = 3 if n_matches > 1 else 1
                    fig_net, ax_net = plt.subplots(figsize=(10, 7))
                    fig_net.set_facecolor('#0e1117')
                    ax_net.set_facecolor('#0e1117')
                    pitch_net = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                    pitch_net.draw(ax=ax_net)
                    if not connections.empty:
                        edges = connections.groupby(['Player', 'NextPlayer']).size().reset_index(name='count')
                        edges = edges[edges['count'] > min_edge_count]
                        max_edge = edges['count'].max() if not edges.empty else 1
                        for _, row in edges.iterrows():
                            p1, p2 = row['Player'], row['NextPlayer']
                            if p1 in avg_pos.index and p2 in avg_pos.index:
                                width = (row['count'] / max_edge) * 6
                                alpha = min(0.9, row['count'] / max_edge + 0.2)
                                pitch_net.lines(avg_pos.loc[p1].x, avg_pos.loc[p1].y,
                                                avg_pos.loc[p2].x, avg_pos.loc[p2].y,
                                                lw=width, color='#ff4b4b', alpha=alpha, ax=ax_net, zorder=1)
                        for player in avg_pos.index:
                            if player in pass_counts.index and player != 'Unknown':
                                s = min(pass_counts[player] * 2, 600)
                                pitch_net.scatter(avg_pos.loc[player].x, avg_pos.loc[player].y,
                                                  s=s, color='#0e1117', edgecolors='white', linewidth=2, ax=ax_net, zorder=2)
                                pitch_net.annotate(player.split(" ")[-1], xy=(avg_pos.loc[player].x, avg_pos.loc[player].y),
                                                   c='white', va='center', ha='center', size=8, ax=ax_net, zorder=3)
                    ax_net.set_title(f'Average Passing Network — All {mgr_short} Matches', color='white', fontsize=12)
                    st.pyplot(fig_net)
                    plt.close(fig_net)

                    st.divider()

                    # --- Average xT per Player ---
                    st.subheader("⚡ Average Expected Threat (xT) per Match")
                    player_xt = utd_df[utd_df['Type'].isin([1, 3])].groupby('Player').agg(
                        xT_Total=('xT_Added', 'sum'),
                        Matches=('Match', 'nunique')
                    ).reset_index()
                    player_xt = player_xt[player_xt['Player'] != 'Unknown']
                    player_xt['xT per Match'] = (player_xt['xT_Total'] / player_xt['Matches']).round(4)
                    player_xt = player_xt.sort_values('xT per Match', ascending=True)

                    fig_xt_bar = px.bar(
                        player_xt, x='xT per Match', y='Player', orientation='h',
                        title='Average xT Created per Match', template='plotly_dark',
                        color='xT per Match', color_continuous_scale=['#ff4b4b', '#00ff85']
                    )
                    fig_xt_bar.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=600)
                    st.plotly_chart(fig_xt_bar, use_container_width=True)
            else:
                st.error(f"Failed to load {mgr_short} match data.")
