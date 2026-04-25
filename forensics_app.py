import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import math
from mplsoccer import Pitch
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull
from io import BytesIO, StringIO
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
            raw = r.content.decode('utf-8-sig', errors='ignore')
            sep = ';' if ';' in raw.split('\n')[0] else ','
            df = pd.read_csv(StringIO(raw), sep=sep)
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


def _calc_xg(x, y, is_header=False):
    """Estimate xG for a shot using a distance-and-angle logistic model.

    Coordinates are Opta 0-100 scale. Pitch dimensions: 105m x 68m.
    Goal at x=100, center y=50, posts at y≈44.6 and y≈55.4.
    """
    x_m = x * 1.05
    y_m = y * 0.68
    goal_x, goal_y = 105.0, 34.0
    post1_y, post2_y = 30.34, 37.66  # 34 ± 3.66

    dist = math.sqrt((goal_x - x_m) ** 2 + (goal_y - y_m) ** 2)
    if dist < 0.5:
        return 0.95

    # Angle subtended by the goal
    dx = max(goal_x - x_m, 0.01)
    v1 = (dx, post1_y - y_m)
    v2 = (dx, post2_y - y_m)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
    angle = math.acos(cos_a)

    if is_header:
        coeff = -2.0 - 0.1 * dist + 1.0 * angle
    else:
        coeff = -1.20 - 0.09 * dist + 1.50 * angle
    return 1.0 / (1.0 + math.exp(-coeff))


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
                        "isThrowIn": 107 in qids,
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

            # Compute xG for shots (Types 13=Miss, 14=Post, 15=SavedShot, 16=Goal)
            is_shot = df_events['Type'].isin([13, 14, 15, 16])
            df_events['xG'] = 0.0
            shot_mask = is_shot & (df_events['Outcome'] != 'Own Goal')
            if shot_mask.any():
                df_events.loc[shot_mask, 'xG'] = df_events.loc[shot_mask].apply(
                    lambda r: _calc_xg(r['x'], r['y']), axis=1
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
        "stats_files": {
            "📊 UnderStat (Amorim & Fletcher)": "fletcher.csv/Amorim and Fletcher UnderStat.csv",
        }
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
        return pd.DataFrame(), {}, {}, []
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
            (full_df['isCorner'] | full_df['isFreeKick'] | full_df['isThrowIn'])
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
            is_throwin = bool(sp_evt.get('isThrowIn', False))
            if is_throwin:
                # Throw-in: only count if no team passes between delivery and goal
                team_passes = between[
                    (between['Team'] == utd_team) & (between['Type'] == 1)
                ]
                if not team_passes.empty:
                    continue
            else:
                # Corner/FK: check for possession-breaking clearing events
                cleared = between[
                    (between['Team'] != utd_team) &
                    (between['Type'].isin([8, 10, 11, 12, 51, 52]))
                ]
                if not cleared.empty:
                    continue
            scorer = g['Player']
            sp_goals[scorer] = sp_goals.get(scorer, 0) + 1
            # The set piece taker gets the assist (they delivered the ball)
            sp_taker = sp_evt['Player']
            if sp_taker != scorer:
                sp_assists[sp_taker] = sp_assists.get(sp_taker, 0) + 1
    # Compute per-match team stats for the Average Team Stats tab
    team_match_stats = []
    utd_team_name = 'Manchester United'
    for match_label, full_df in full_dfs.items():
        utd_events = full_df[(full_df['Team'] == utd_team_name) & (full_df['Player'] != 'Unknown')]
        opp_events = full_df[(full_df['Team'] != utd_team_name) & (full_df['Player'] != 'Unknown')]
        utd_passes = len(utd_events[utd_events['Type'] == 1])
        opp_passes = len(opp_events[opp_events['Type'] == 1])
        possession = round(utd_passes / (utd_passes + opp_passes) * 100, 1) if (utd_passes + opp_passes) > 0 else 0
        utd_shots = utd_events[utd_events['Type'].isin([13, 14, 15, 16]) & (utd_events['Outcome'] != 'Own Goal')]
        match_xg = round(utd_shots['xG'].sum(), 2)
        utd_f3 = len(utd_events[(utd_events['Type'] == 1) & (utd_events['x'] > 66.6)])
        opp_f3 = len(opp_events[(opp_events['Type'] == 1) & (opp_events['x'] > 66.6)])
        total_f3 = utd_f3 + opp_f3
        field_tilt = round(utd_f3 / total_f3 * 100, 1) if total_f3 > 0 else 0
        utd_def = len(utd_events[utd_events['Type'].isin([4, 7, 8])])
        ppda = round(opp_passes / utd_def, 1) if utd_def > 0 else 0
        recoveries = utd_events[
            ((utd_events['Type'] == 7) & (utd_events['Outcome'] == 'Successful')) |
            (utd_events['Type'].isin([8, 12]))
        ]
        avg_def_line = round(recoveries['x'].mean(), 1) if not recoveries.empty else 0
        team_match_stats.append({
            'Match': match_label, 'Possession': possession, 'xG': match_xg,
            'Field Tilt': field_tilt, 'PPDA': ppda, 'Def Line': avg_def_line
        })

    return combined, sp_goals, sp_assists, team_match_stats


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
            sp_evt = prior.iloc[-1]
            sp_idx = sp_evt['Index']
            between = match_df[
                (match_df['Index'] > sp_idx) & (match_df['Index'] < g['Index'])
            ]
            is_throwin = bool(sp_evt.get('isThrowIn', False))
            if is_throwin:
                # Throw-in: only count if no team passes between delivery and goal
                team_passes = between[
                    (between['Team'] == team) & (between['Type'] == 1)
                ]
                if team_passes.empty:
                    goal_indices.add(sp_idx)
            else:
                # Corner/FK: check for possession-breaking clearing events
                cleared = between[
                    (between['Team'] != team) &
                    (between['Type'].isin([8, 10, 11, 12, 51, 52]))
                ]
                if cleared.empty:
                    goal_indices.add(sp_idx)

    for _, p in sp_df.iterrows():
        if p['Index'] in goal_indices:
            goal_sp.append(p)
        else:
            reg_sp.append(p)
    return pd.DataFrame(goal_sp), pd.DataFrame(reg_sp)


def _fix_gk_positions(def_avg, source_df):
    """Replace goalkeeper avg positions with their GK-event averages and
    fill in missing outfield players using their all-event averages.

    Keepers are identified by having save events (Type 10). Their defensive
    position is recalculated from GK-specific events (saves, claims, pick-ups,
    punches, goal kicks) which better represent their actual pitch position.

    Outfield players active in the time range but without defensive events
    are added at their average position across all events so that the full
    squad is visible.
    """
    gk_types = [10, 11, 50, 51, 52]
    keepers = set(source_df[source_df['Type'] == 10]['Player'].unique())
    for gk in keepers:
        gk_events = source_df[(source_df['Player'] == gk) & (source_df['Type'].isin(gk_types))]
        if not gk_events.empty:
            if gk in def_avg.index:
                def_avg.loc[gk, 'x'] = gk_events['x'].mean()
                def_avg.loc[gk, 'y'] = gk_events['y'].mean()
            else:
                def_avg.loc[gk] = {'x': gk_events['x'].mean(), 'y': gk_events['y'].mean()}
    # Fill missing outfield players from all-event averages
    all_players = source_df[source_df['Player'] != 'Unknown']['Player'].unique()
    for player in all_players:
        if player not in def_avg.index and player not in keepers:
            p_events = source_df[source_df['Player'] == player]
            if not p_events.empty:
                def_avg.loc[player] = {'x': p_events['x'].mean(), 'y': p_events['y'].mean()}
    return def_avg


def _make_plotly_pitch(title):
    """Create a reusable Opta-like pitch in Plotly coordinates (0-100)."""
    line = dict(color="white", width=1.5)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=line),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=line),
        dict(type="circle", x0=40, y0=40, x1=60, y1=60, line=line),
        dict(type="rect", x0=0, y0=21.1, x1=17, y1=78.9, line=line),
        dict(type="rect", x0=83, y0=21.1, x1=100, y1=78.9, line=line),
        dict(type="rect", x0=0, y0=36.8, x1=5.8, y1=63.2, line=line),
        dict(type="rect", x0=94.2, y0=36.8, x1=100, y1=63.2, line=line),
    ]

    fig = go.Figure()
    fig.update_layout(
        title=title,
        shapes=shapes,
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=0.68),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        margin=dict(l=10, r=10, t=45, b=10),
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                    bgcolor='rgba(14,17,23,0.7)', bordercolor='#444', borderwidth=1)
    )
    # Attack direction arrow (left goal → right goal)
    fig.add_annotation(
        x=88, y=-1.5, ax=12, ay=-1.5,
        text="►  Attack Direction",
        showarrow=True, arrowhead=4, arrowwidth=3,
        arrowcolor='#ffd700',
        font=dict(color='#ffd700', size=12, family='Arial Black'),
        xref='x', yref='y', axref='x', ayref='y',
    )
    return fig


def _add_plotly_action_lines(fig, df, name, color, dash='solid', width=2, marker_symbol='circle', marker_size=8, arrows=False):
    """Draw action vectors plus event-end markers with player/minute hover.
    Pass arrows=True for pass maps to show direction with arrowhead markers."""
    if df.empty:
        return

    xs, ys = [], []
    for _, r in df.iterrows():
        xs.extend([r['x'], r['endX'], None])
        ys.extend([r['y'], r['endY'], None])

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        hoverinfo='skip',
        legendgroup=name,
    ))

    custom = np.column_stack([
        df['Player'].astype(str),
        df['Minute'].astype(int),
        df['Outcome'].astype(str),
    ])

    if arrows:
        # Place an arrowhead marker 75% along each pass, angled in direction of travel
        _dx = (df['endX'] - df['x']).values
        _dy = (df['endY'] - df['y']).values
        _arrow_x = df['x'].values + 0.75 * _dx
        _arrow_y = df['y'].values + 0.75 * _dy
        # Plotly arrow marker: 0 = north, clockwise. atan2 gives CCW from east.
        _angles = (90 - np.degrees(np.arctan2(_dy, _dx))) % 360
        fig.add_trace(go.Scatter(
            x=_arrow_x,
            y=_arrow_y,
            mode='markers',
            marker=dict(color=color, size=marker_size + 3, symbol='arrow',
                        angle=_angles, line=dict(color=color, width=1)),
            name=f"{name} Dir",
            showlegend=False,
            legendgroup=name,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Minute: %{customdata[1]}'<br>"
                "Outcome: %{customdata[2]}<extra></extra>"
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['endX'],
            y=df['endY'],
            mode='markers',
            marker=dict(color=color, size=marker_size, symbol=marker_symbol, line=dict(color='white', width=1)),
            name=f"{name} End",
            showlegend=False,
            legendgroup=name,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Minute: %{customdata[1]}'<br>"
                "Outcome: %{customdata[2]}<extra></extra>"
            ),
        ))


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
        sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["📊 STATISTICAL REPORTS", "⚽ MATCH TELEMETRY", "👥 AVERAGE PLAYER STATS", "📈 AVERAGE TEAM STATS"])

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
                            display_df.index = range(1, len(display_df) + 1)
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
                        plot_df = plot_df[plot_df['Player'] != 'Unknown']

                        if sel_player != "All Players":
                            plot_df = plot_df[plot_df['Player'] == sel_player]

                        opp_team = [t for t in teams if t != sel_team][0]
                        opp_stats = match_df[match_df['Team'] == opp_team].copy()
                        opp_stats = opp_stats[(opp_stats['Minute'] >= min_range[0]) & (opp_stats['Minute'] <= min_range[1])]
                        opp_stats = opp_stats[opp_stats['Player'] != 'Unknown']

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
                        goals = len(plot_df[(plot_df['Type'] == 16) & (plot_df['Outcome'] != 'Own Goal')])
                        own_goals = len(plot_df[(plot_df['Type'] == 16) & (plot_df['Outcome'] == 'Own Goal')])
                        tackles = len(plot_df[plot_df['Type'] == 7])
                        total_def = tackles + len(plot_df[plot_df['Type'] == 8]) + len(plot_df[plot_df['Type'] == 12])
                        fouls = len(plot_df[plot_df['Type'] == 4])

                        recoveries = plot_df[
                            ((plot_df['Type'] == 7) & (plot_df['Outcome'] == 'Successful')) |
                            (plot_df['Type'].isin([8, 12]))
                        ]
                        avg_rec_height = round(recoveries['x'].mean(), 1) if not recoveries.empty else 0

                        # Possession %
                        opp_total_passes = len(opp_stats[opp_stats['Type'] == 1])
                        possession = round(total_passes / (total_passes + opp_total_passes) * 100, 1) if (total_passes + opp_total_passes) > 0 else 0

                        # Expected Goals
                        team_xg = round(plot_df.loc[plot_df['Type'].isin([13, 14, 15, 16]) & (plot_df['Outcome'] != 'Own Goal'), 'xG'].sum(), 2)

                        st.divider()
                        m1, m2, m3, m4, m5, m6, m7, m8, m9 = st.columns(9)
                        m1.metric("⚽ Passing", f"{succ_passes}/{total_passes}", f"{pass_acc}%")
                        m2.metric("🎯 Shooting", f"{goals} Goals", f"{total_shots} Shots" + (f" · {own_goals} Own Goal{'s' if own_goals > 1 else ''}" if own_goals else ""))
                        m3.metric("📊 Possession", f"{possession}%", "Pass-Based")
                        m4.metric("📈 xG", f"{team_xg}", f"{goals} Actual Goals")
                        m5.metric("🛡️ Def. Actions", f"{total_def}", f"{tackles} Tackles")
                        m6.metric("⚠️ Discipline", f"{fouls} Fouls", "Committed")
                        m7.metric("⚖️ Field Tilt", f"{field_tilt}%", "Final 3rd Share")
                        m8.metric("🛑 PPDA", f"{ppda}", "Passes per Def. Action")
                        m9.metric("📏 Def. Line", f"{avg_rec_height}m", "Avg Recovery Height")
                        st.divider()

                        # --- Team progression insights ---
                        prog_passes = plot_df[
                            (plot_df['Type'] == 1) &
                            (plot_df['Outcome'] == 'Successful') &
                            ((plot_df['endX'] - plot_df['x']) >= 10)
                        ]
                        prog_leaders = prog_passes.groupby('Player').size().reset_index(name='Progressive Passes').sort_values('Progressive Passes', ascending=False)

                        st.divider()
                        st.markdown("#### 🗂️ Select Evidence Layers")

                        in_pos_options = [
                            "Progressive Passes Map", "Pass Map", "Passing Heatmap",
                            "Expected Threat (xT) Grid", "High Value Pass Map",
                            "Match Progression",
                            "The Architect (Build-Up Phase)", "Passing Network (Structure)"
                        ]
                        out_pos_options = [
                            "Defensive Actions Map", "Defensive Shield (Heatmap + Line)"
                        ]
                        attacking_phase_options = [
                            "Actions Leading to Shots",
                            "Zone 14 & Half-Spaces", "Zone Invasions",
                            "Average Attacking & Defending Positions"
                        ]
                        att_trans_options = [
                            "Progression Trajectory Lines", "Time-to-Shot Scatter Plot", "Transition Map"
                        ]
                        def_trans_options = [
                            "Recovery vs. Loss Maps", "Defensive Reaction Time/Distance Curves"
                        ]
                        set_options = [
                            "Set Piece Targeting (Corners)", "Free Kick Targeting"
                        ]
                        gk_options = [
                            "Shot Trajectory Map (GK View)", "Goal Kick Direction Map"
                        ]
                        row1_c1, row1_c2, row1_c3 = st.columns(3)
                        with row1_c1:
                            in_pos_mods = st.multiselect("🟢 In-Possession", in_pos_options, default=["Progressive Passes Map"], key=f"mip_{manager}")
                        with row1_c2:
                            out_pos_mods = st.multiselect("🔴 Out of Possession", out_pos_options, key=f"mop_{manager}")
                        with row1_c3:
                            attacking_phase_mods = st.multiselect("⚔️ Attacking Phase", attacking_phase_options, key=f"map_{manager}")

                        row2_c1, row2_c2, row2_c3 = st.columns(3)
                        with row2_c1:
                            att_trans_mods = st.multiselect("⚡ Attacking Transition", att_trans_options, key=f"mat_{manager}")
                        with row2_c2:
                            def_trans_mods = st.multiselect("🧯 Defensive Transition", def_trans_options, key=f"mdt_{manager}")
                        with row2_c3:
                            set_mods = st.multiselect("🎯 Set-Pieces", set_options, key=f"msp_{manager}")

                        row3_c1, _row3_c2, _row3_c3 = st.columns(3)
                        with row3_c1:
                            gk_mods = st.multiselect("🧤 Goalkeeping", gk_options, key=f"mgk_{manager}")

                        modules = in_pos_mods + out_pos_mods + attacking_phase_mods + att_trans_mods + def_trans_mods + set_mods + gk_mods

                        # Reusable transition pairing for defensive-transition modules
                        losses_base = plot_df[(plot_df['Type'].isin([1, 3])) & (plot_df['Outcome'] == 'Unsuccessful')].sort_values('Index')
                        transitions = []
                        for _, loss in losses_base.iterrows():
                            rec = match_df[
                                (match_df['Index'] > loss['Index']) &
                                (match_df['Index'] <= loss['Index'] + 25) &
                                (match_df['Team'] == sel_team) &
                                (match_df['Type'].isin([7, 8, 12]))
                            ].sort_values('Index').head(1)
                            if rec.empty:
                                continue
                            r = rec.iloc[0]
                            transitions.append({
                                'loss_x': loss['x'], 'loss_y': loss['y'], 'loss_player': loss['Player'], 'loss_min': int(loss['Minute']), 'loss_idx': int(loss['Index']),
                                'rec_x': r['x'], 'rec_y': r['y'], 'rec_player': r['Player'], 'rec_min': int(r['Minute']), 'rec_idx': int(r['Index'])
                            })
                        transitions_df = pd.DataFrame(transitions)

                        # --- In-Possession ---
                        if "Progressive Passes Map" in modules:
                            st.subheader("🟢 Progressive Passes Map")
                            _n_pp = len(prog_passes)
                            _pp_ply = prog_passes['Player'].nunique() if not prog_passes.empty else 0
                            _pp_dist = round((prog_passes['endX'] - prog_passes['x']).mean(), 1) if not prog_passes.empty else 0
                            _mc1, _mc2, _mc3 = st.columns(3)
                            _mc1.metric("Progressive Passes", _n_pp)
                            _mc2.metric("Players Involved", _pp_ply)
                            _mc3.metric("Avg Forward Distance", f"{_pp_dist} u")
                            fig_pp = _make_plotly_pitch(f"{sel_team} Progressive Passes")
                            _add_plotly_action_lines(fig_pp, prog_passes, "Progressive Pass", "#00ff85", width=2, arrows=True)
                            st.plotly_chart(fig_pp, use_container_width=True)
                            if not prog_leaders.empty:
                                with st.expander("👥 Progressive Pass Leaders"):
                                    fig_prog_lead = px.bar(prog_leaders.head(8).sort_values('Progressive Passes'), x='Progressive Passes', y='Player', orientation='h', template='plotly_dark')
                                    fig_prog_lead.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                    st.plotly_chart(fig_prog_lead, use_container_width=True)

                        if "Pass Map" in modules:
                            st.subheader("🟢 Pass Map")
                            succ_passes_df = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful')]
                            fail_passes_df = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Unsuccessful')]
                            _n_succ = len(succ_passes_df); _n_fail = len(fail_passes_df)
                            _acc = round(_n_succ / (_n_succ + _n_fail) * 100, 1) if (_n_succ + _n_fail) > 0 else 0
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("✅ Completed", _n_succ)
                            _mc2.metric("❌ Incomplete", _n_fail)
                            _mc3.metric("Accuracy", f"{_acc}%")
                            _mc4.metric("Players", succ_passes_df['Player'].nunique() if not succ_passes_df.empty else 0)
                            fig_pm = _make_plotly_pitch("Completed vs Failed Passes")
                            _add_plotly_action_lines(fig_pm, succ_passes_df, "✅ Successful", "#00ff85", width=2, arrows=True)
                            _add_plotly_action_lines(fig_pm, fail_passes_df, "❌ Unsuccessful", "#ff4b4b", width=2, dash='dot', arrows=True)
                            st.plotly_chart(fig_pm, use_container_width=True)
                            pass_leaders = succ_passes_df.groupby('Player').size().reset_index(name='Completed Passes').sort_values('Completed Passes', ascending=False)
                            fail_leaders = fail_passes_df.groupby('Player').size().reset_index(name='Unsuccessful Passes').sort_values('Unsuccessful Passes', ascending=False)
                            _exp_c1, _exp_c2 = st.columns(2)
                            with _exp_c1:
                                if not pass_leaders.empty:
                                    with st.expander("👥 Pass Completion Leaders"):
                                        fig_pl = px.bar(pass_leaders.head(8).sort_values('Completed Passes'), x='Completed Passes', y='Player', orientation='h', template='plotly_dark')
                                        fig_pl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_pl, use_container_width=True)
                            with _exp_c2:
                                if not fail_leaders.empty:
                                    with st.expander("❌ Unsuccessful Pass Leaders"):
                                        fig_fl = px.bar(fail_leaders.head(8).sort_values('Unsuccessful Passes'), x='Unsuccessful Passes', y='Player', orientation='h', template='plotly_dark', color_discrete_sequence=['#ff4b4b'])
                                        fig_fl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_fl, use_container_width=True)

                        if "Passing Heatmap" in modules:
                            st.subheader("🟢 Passing Heatmap")
                            pass_heat = viz_df[viz_df['Type'] == 1]
                            if not pass_heat.empty:
                                _n_ph = len(pass_heat); _ph_ply = pass_heat['Player'].nunique()
                                _ph_succ = len(pass_heat[pass_heat['Outcome'] == 'Successful'])
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("Total Passes", _n_ph)
                                _mc2.metric("Completed", _ph_succ)
                                _mc3.metric("Players", _ph_ply)
                                fig_ph = _make_plotly_pitch("Pass Origin Density — brighter = more passes")
                                fig_ph.add_trace(go.Histogram2d(
                                    x=pass_heat['x'], y=pass_heat['y'], nbinsx=20, nbinsy=14,
                                    colorscale='Turbo', reversescale=False, opacity=0.75,
                                    showscale=True, colorbar=dict(title='Pass Count')
                                ))
                                st.plotly_chart(fig_ph, use_container_width=True)
                                pass_vol_leaders = pass_heat.groupby('Player').size().reset_index(name='Passes').sort_values('Passes', ascending=False)
                                if not pass_vol_leaders.empty:
                                    with st.expander("👥 Pass Volume Leaders"):
                                        fig_pvl = px.bar(pass_vol_leaders.head(8).sort_values('Passes'), x='Passes', y='Player', orientation='h', template='plotly_dark')
                                        fig_pvl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_pvl, use_container_width=True)
                            else:
                                st.info("No passes in this minute range.")

                        if "Expected Threat (xT) Grid" in modules:
                            st.subheader("🟢 Expected Threat (xT) Grid")
                            xt_acts = viz_df[(viz_df['Type'].isin([1, 3])) & (viz_df['Outcome'] == 'Successful') & (viz_df['xT_Added'] > 0)]
                            if not xt_acts.empty:
                                _xt_total = round(xt_acts['xT_Added'].sum(), 3)
                                _xt_n = len(xt_acts); _xt_ply = xt_acts['Player'].nunique()
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("Total xT Added", _xt_total)
                                _mc2.metric("Actions", _xt_n)
                                _mc3.metric("Players", _xt_ply)
                                fig_l_xt = _make_plotly_pitch("xT Grid — cell value = xT generated, brighter = higher threat")
                                xbins = np.linspace(0, 100, 13)
                                ybins = np.linspace(0, 100, 9)
                                z, xedges, yedges = np.histogram2d(
                                    xt_acts['x'].values,
                                    xt_acts['y'].values,
                                    bins=[xbins, ybins],
                                    weights=xt_acts['xT_Added'].values
                                )
                                x_centers = (xedges[:-1] + xedges[1:]) / 2
                                y_centers = (yedges[:-1] + yedges[1:]) / 2
                                fig_l_xt.add_trace(go.Heatmap(
                                    x=x_centers,
                                    y=y_centers,
                                    z=z.T,
                                    colorscale='Magma',
                                    opacity=0.75,
                                    showscale=True,
                                    colorbar=dict(title='xT')
                                ))
                                for xi, xv in enumerate(x_centers):
                                    for yi, yv in enumerate(y_centers):
                                        val = z[xi, yi]
                                        if val > 0.001:
                                            fig_l_xt.add_annotation(
                                                x=float(xv), y=float(yv),
                                                text=f"{val:.3f}",
                                                showarrow=False,
                                                font=dict(color='white', size=9)
                                            )
                                st.plotly_chart(fig_l_xt, use_container_width=True)
                                xt_leaders = xt_acts.groupby('Player')['xT_Added'].sum().reset_index(name='xT Added').sort_values('xT Added', ascending=False)
                                if not xt_leaders.empty:
                                    with st.expander("👥 xT Leaders"):
                                        fig_xtl = px.bar(xt_leaders.head(8).sort_values('xT Added'), x='xT Added', y='Player', orientation='h', template='plotly_dark')
                                        fig_xtl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_xtl, use_container_width=True)
                            else:
                                st.info("No xT values available for current filters.")

                        if "High Value Pass Map" in modules:
                            st.subheader("🟢 High Value Pass Map")
                            xT_thresh = viz_df.loc[viz_df['xT_Added'] > 0, 'xT_Added'].quantile(0.75) if (viz_df['xT_Added'] > 0).any() else 0.05
                            high_xt = viz_df[(viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful') & (viz_df['xT_Added'] >= xT_thresh)]
                            shot_idxs = viz_df[viz_df['Type'].isin([13, 14, 15, 16])]['Index'].values
                            pre_shot_rows = []
                            for si in shot_idxs:
                                w = viz_df[(viz_df['Index'].between(si - 4, si - 1)) & (viz_df['Type'] == 1) & (viz_df['Outcome'] == 'Successful')]
                                if not w.empty:
                                    pre_shot_rows.append(w.iloc[-1])
                            pre_shot_df = pd.DataFrame(pre_shot_rows).drop_duplicates(subset=['Index']) if pre_shot_rows else pd.DataFrame()
                            if not high_xt.empty or not pre_shot_df.empty:
                                _n_hxt = len(high_xt); _n_ps = len(pre_shot_df)
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("🟡 High xT Passes", _n_hxt, help="Top 25% by xT value")
                                _mc2.metric("🟠 Pre-Shot Passes", _n_ps)
                                _mc3.metric("xT Threshold", f"{round(xT_thresh, 4)}")
                                fig_hvp = _make_plotly_pitch("High Value Passes — 🟡 High xT · 🟠 Pre-Shot")
                                if not high_xt.empty:
                                    _add_plotly_action_lines(fig_hvp, high_xt, "🟡 High xT Pass", "#ffd700", width=2, arrows=True)
                                if not pre_shot_df.empty:
                                    _add_plotly_action_lines(fig_hvp, pre_shot_df, "🟠 Pre-Shot Pass", "#ff7f50", width=3, arrows=True)
                                st.plotly_chart(fig_hvp, use_container_width=True)
                                all_hv = pd.concat([high_xt, pre_shot_df]).drop_duplicates(subset=['Index']) if not pre_shot_df.empty else high_xt
                                hv_leaders = all_hv.groupby('Player').size().reset_index(name='High Value Passes').sort_values('High Value Passes', ascending=False)
                                if not hv_leaders.empty:
                                    with st.expander("👥 High Value Pass Leaders"):
                                        fig_hvl = px.bar(hv_leaders.head(8).sort_values('High Value Passes'), x='High Value Passes', y='Player', orientation='h', template='plotly_dark')
                                        fig_hvl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_hvl, use_container_width=True)
                            else:
                                st.info("No high-value passes found in this range.")

                        if "Average Attacking & Defending Positions" in modules:
                            st.subheader("🟢 Average Attacking Positions vs. Opposition's Average Defensive Positions")
                            att_evts = viz_df[viz_df['Type'].isin([1, 3])]
                            # Opposition defensive actions — flip x so both teams face same direction
                            opp_def_raw = match_df[(match_df['Team'] == opp_team) & (match_df['Type'].isin([4, 7, 8, 12]))].copy()
                            opp_def_raw['x'] = 100 - opp_def_raw['x']
                            opp_def_raw['y'] = 100 - opp_def_raw['y']
                            # Apply same minute filter as viz_df
                            opp_def_evts = opp_def_raw[(opp_def_raw['Minute'] >= min_range[0]) & (opp_def_raw['Minute'] <= min_range[1])]
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric(f"🟢 {sel_team} Att. Actions", len(att_evts))
                            _mc2.metric(f"🔴 {opp_team} Def. Actions", len(opp_def_evts))
                            _mc3.metric(f"{sel_team} Players", att_evts['Player'].nunique() if not att_evts.empty else 0)
                            _mc4.metric(f"{opp_team} Players", opp_def_evts['Player'].nunique() if not opp_def_evts.empty else 0)
                            fig_avgpos = _make_plotly_pitch(f"{sel_team} ⚔ Attack  vs.  {opp_team} 🛡 Defence")
                            if not att_evts.empty:
                                att_avg = att_evts.groupby('Player')[['x', 'y']].mean().reset_index()
                                att_cnts = att_evts.groupby('Player').size().reset_index(name='Actions')
                                att_avg = att_avg.merge(att_cnts, on='Player')
                                node_s = np.clip(att_avg['Actions'] * 1.5, 10, 35)
                                fig_avgpos.add_trace(go.Scatter(
                                    x=att_avg['x'], y=att_avg['y'], mode='markers+text',
                                    name=f'🟢 {sel_team} Attack',
                                    marker=dict(size=node_s, color='#00ff85', opacity=0.85, line=dict(color='white', width=1.5)),
                                    text=att_avg['Player'].str.split(' ').str[-1],
                                    textposition='top center', textfont=dict(color='#00ff85', size=9),
                                    customdata=np.column_stack([att_avg['Player'], att_avg['Actions']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Avg Attack Pos<br>Actions: %{customdata[1]}<extra></extra>"
                                ))
                            if not opp_def_evts.empty:
                                def_avg = opp_def_evts.groupby('Player')[['x', 'y']].mean().reset_index()
                                def_cnts = opp_def_evts.groupby('Player').size().reset_index(name='Actions')
                                def_avg = def_avg.merge(def_cnts, on='Player')
                                node_sd = np.clip(def_avg['Actions'] * 2, 10, 35)
                                fig_avgpos.add_trace(go.Scatter(
                                    x=def_avg['x'], y=def_avg['y'], mode='markers+text',
                                    name=f'🔴 {opp_team} Defence',
                                    marker=dict(size=node_sd, color='#ff4b4b', opacity=0.85, symbol='diamond', line=dict(color='white', width=1.5)),
                                    text=def_avg['Player'].str.split(' ').str[-1],
                                    textposition='bottom center', textfont=dict(color='#ff4b4b', size=9),
                                    customdata=np.column_stack([def_avg['Player'], def_avg['Actions']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Avg Defend Pos<br>Actions: %{customdata[1]}<extra></extra>"
                                ))
                            st.plotly_chart(fig_avgpos, use_container_width=True)
                            _exp1, _exp2 = st.columns(2)
                            with _exp1:
                                if not att_evts.empty:
                                    att_act_leaders = att_evts.groupby('Player').size().reset_index(name='Attacking Actions').sort_values('Attacking Actions', ascending=False)
                                    with st.expander(f"👥 {sel_team} Attacking Leaders"):
                                        fig_aal = px.bar(att_act_leaders.head(8).sort_values('Attacking Actions'), x='Attacking Actions', y='Player', orientation='h', template='plotly_dark')
                                        fig_aal.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_aal, use_container_width=True)
                            with _exp2:
                                if not opp_def_evts.empty:
                                    opp_def_leaders = opp_def_evts.groupby('Player').size().reset_index(name='Defensive Actions').sort_values('Defensive Actions', ascending=False)
                                    with st.expander(f"👥 {opp_team} Defensive Leaders"):
                                        fig_odl = px.bar(opp_def_leaders.head(8).sort_values('Defensive Actions'), x='Defensive Actions', y='Player', orientation='h', template='plotly_dark', color_discrete_sequence=['#ff4b4b'])
                                        fig_odl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_odl, use_container_width=True)

                        if "Match Progression" in modules:
                            st.subheader("📈 Match Progression")
                            all_match_shots = match_df[match_df['Type'].isin([13, 14, 15, 16])].sort_values('Minute')
                            team_shots_mp = all_match_shots[all_match_shots['Team'] == sel_team].copy()
                            opp_shots_mp = all_match_shots[all_match_shots['Team'] != sel_team].copy()
                            _t_xg = round(team_shots_mp['xG'].sum(), 2); _o_xg = round(opp_shots_mp['xG'].sum(), 2)
                            _t_goals = len(team_shots_mp[team_shots_mp['Type'] == 16]); _o_goals = len(opp_shots_mp[opp_shots_mp['Type'] == 16])
                            _t_shots = len(team_shots_mp); _o_shots = len(opp_shots_mp)
                            _mc1, _mc2, _mc3, _mc4, _mc5, _mc6 = st.columns(6)
                            _mc1.metric(f"{sel_team} xG", _t_xg)
                            _mc2.metric(f"{opp_team} xG", _o_xg)
                            _mc3.metric(f"{sel_team} Goals", _t_goals)
                            _mc4.metric(f"{opp_team} Goals", _o_goals)
                            _mc5.metric(f"{sel_team} Shots", _t_shots)
                            _mc6.metric(f"{opp_team} Shots", _o_shots)
                            mins_mp = list(range(0, max_minute + 2))
                            team_cum, opp_cum = [], []
                            rt, ro = 0.0, 0.0
                            for m in mins_mp:
                                rt += team_shots_mp.loc[team_shots_mp['Minute'] == m, 'xG'].sum()
                                ro += opp_shots_mp.loc[opp_shots_mp['Minute'] == m, 'xG'].sum()
                                team_cum.append(round(rt, 3))
                                opp_cum.append(round(ro, 3))
                            fig_mp = go.Figure()
                            fig_mp.add_trace(go.Scatter(x=mins_mp, y=team_cum, mode='lines', line=dict(color='#00ff85', width=3), fill='tozeroy', fillcolor='rgba(0,255,133,0.10)', name=f'{sel_team} xG'))
                            fig_mp.add_trace(go.Scatter(x=mins_mp, y=opp_cum, mode='lines', line=dict(color='#ff4b4b', width=3), fill='tozeroy', fillcolor='rgba(255,75,75,0.10)', name=f'{opp_team} xG'))
                            for _, g in team_shots_mp[team_shots_mp['Type'] == 16].iterrows():
                                m_idx = int(min(g['Minute'], max_minute))
                                fig_mp.add_vline(x=g['Minute'], line_color='#00ff85', line_dash='dot', line_width=1.5)
                                fig_mp.add_annotation(x=g['Minute'], y=team_cum[m_idx], text=f"⚽ {str(g['Player']).split(' ')[-1]}", font=dict(color='#00ff85', size=10), showarrow=False, yshift=14)
                            for _, g in opp_shots_mp[opp_shots_mp['Type'] == 16].iterrows():
                                m_idx = int(min(g['Minute'], max_minute))
                                fig_mp.add_vline(x=g['Minute'], line_color='#ff4b4b', line_dash='dot', line_width=1.5)
                                fig_mp.add_annotation(x=g['Minute'], y=opp_cum[m_idx], text=f"⚽ {str(g['Player']).split(' ')[-1]}", font=dict(color='#ff4b4b', size=10), showarrow=False, yshift=-14)
                            fig_mp.update_layout(template='plotly_dark', title='Match Progression — Cumulative xG', xaxis_title='Minute', yaxis_title='Cumulative xG', hovermode='x unified', height=420)
                            st.plotly_chart(fig_mp, use_container_width=True)
                            xg_leaders_mp = team_shots_mp.groupby('Player')['xG'].sum().reset_index(name='xG').sort_values('xG', ascending=False)
                            if not xg_leaders_mp.empty:
                                with st.expander("👥 xG Leaders"):
                                    fig_mpl = px.bar(xg_leaders_mp.head(8).sort_values('xG'), x='xG', y='Player', orientation='h', template='plotly_dark')
                                    fig_mpl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                    st.plotly_chart(fig_mpl, use_container_width=True)

                        if "The Architect (Build-Up Phase)" in modules:
                            st.subheader("🧭 The Architect (Build-Up Phase)")
                            build_up = viz_df[(viz_df['Type'] == 1) & (viz_df['x'] < 33)].copy()
                            if not build_up.empty:
                                build_up['dist'] = np.sqrt(
                                    (build_up['endX'] - build_up['x'])**2 + (build_up['endY'] - build_up['y'])**2
                                )
                                build_up['Length'] = np.where(build_up['dist'] > 25, 'Long', 'Short')
                                build_up['Outcome'] = build_up['Outcome'].fillna('Unknown')
                                _successful = build_up[build_up['Outcome'] == 'Successful']
                                _unsuccessful = build_up[build_up['Outcome'] != 'Successful']
                                _short = _successful[_successful['Length'] == 'Short']
                                _long = _successful[_successful['Length'] == 'Long']
                                build_up['TargetThird'] = pd.cut(
                                    build_up['endX'], bins=[0, 33, 66, 100],
                                    labels=['Own Third', 'Middle Third', 'Attacking Third'], include_lowest=True
                                )
                                build_up['TargetChannel'] = pd.cut(
                                    build_up['endY'], bins=[0, 33, 67, 100],
                                    labels=['Left', 'Center', 'Right'], include_lowest=True
                                )
                                build_up['Zone'] = build_up['TargetThird'].astype(str) + ' — ' + build_up['TargetChannel'].astype(str)
                                _pass_pct = round(len(_successful) / len(build_up) * 100, 1) if len(build_up) > 0 else 0
                                _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
                                _mc1.metric("Build-Up Passes", len(build_up))
                                _mc2.metric("✅ Successful", len(_successful))
                                _mc3.metric("❌ Unsuccessful", len(_unsuccessful))
                                _mc4.metric("Completion %", f"{_pass_pct}%")
                                _mc5.metric("Players", build_up['Player'].nunique())
                                fig_arch = _make_plotly_pitch("The Architect — 🔵 Short · 🟡 Long · 🔴 Lost")
                                # Zone grid dividers
                                for xv in [33, 66]:
                                    fig_arch.add_shape(type='line', x0=xv, y0=0, x1=xv, y1=100,
                                                       line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=1.5))
                                for yv in [33, 67]:
                                    fig_arch.add_shape(type='line', x0=0, y0=yv, x1=100, y1=yv,
                                                       line=dict(color='rgba(255,255,255,0.2)', dash='dot', width=1))
                                # Third labels along top
                                for _lbl, _xc in [('Own ⅓', 16.5), ('Middle ⅓', 49.5), ('Att ⅓', 83)]:
                                    fig_arch.add_annotation(x=_xc, y=97, text=_lbl, showarrow=False,
                                                            font=dict(color='rgba(255,255,255,0.5)', size=10))
                                # Channel labels on left edge
                                for _lbl, _yc in [('Left', 16.5), ('Center', 50), ('Right', 83.5)]:
                                    fig_arch.add_annotation(x=1, y=_yc, text=_lbl, showarrow=False,
                                                            font=dict(color='rgba(255,255,255,0.4)', size=9), xanchor='left')
                                if not _short.empty:
                                    _add_plotly_action_lines(fig_arch, _short, "🔵 Short Pass", "#00ffff", width=2, arrows=True)
                                if not _long.empty:
                                    _add_plotly_action_lines(fig_arch, _long, "🟡 Long Pass", "#ffd700", width=2, arrows=True)
                                if not _unsuccessful.empty:
                                    _add_plotly_action_lines(fig_arch, _unsuccessful, "🔴 Lost Pass", "#ff4b4b", width=2, arrows=True)
                                st.plotly_chart(fig_arch, use_container_width=True)
                                zone_counts = build_up.groupby('Zone').size().reset_index(name='Passes').sort_values('Passes', ascending=False)
                                with st.expander("📊 Zone Breakdown & Build-Up Leaders"):
                                    _zc1, _zc2 = st.columns(2)
                                    with _zc1:
                                        st.markdown("**Passes by Target Zone**")
                                        fig_zb = px.bar(zone_counts, x='Passes', y='Zone', orientation='h',
                                                        template='plotly_dark', color='Passes',
                                                        color_continuous_scale='Turbo')
                                        fig_zb.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                                             showlegend=False, coloraxis_showscale=False)
                                        st.plotly_chart(fig_zb, use_container_width=True)
                                    with _zc2:
                                        st.markdown("**Top Players (Build-Up Volume)**")
                                        _build_leaders = build_up.groupby('Player').size().reset_index(name='Build-Up Passes').sort_values('Build-Up Passes', ascending=False)
                                        fig_bul = px.bar(_build_leaders.head(8).sort_values('Build-Up Passes'),
                                                         x='Build-Up Passes', y='Player', orientation='h', template='plotly_dark')
                                        fig_bul.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                                        st.plotly_chart(fig_bul, use_container_width=True)
                            else:
                                st.info("No build-up pass actions available for current filters.")

                        if "Passing Network (Structure)" in modules:
                            st.subheader("🧭 Passing Network (Structure)")
                            st.caption("Node size = event involvement · Edge thickness = pass frequency between two players (min 3 passes)")
                            net_df = viz_df.copy()
                            avg_pos = net_df.groupby('Player')[['x', 'y']].mean()
                            pass_counts = net_df.groupby('Player').size()
                            net_df['NextPlayer'] = net_df['Player'].shift(-1)
                            net_df['NextTeam'] = net_df['Team'].shift(-1)
                            edges = net_df[(net_df['Type'] == 1) & (net_df['Outcome'] == 'Successful') & (net_df['NextTeam'] == sel_team)]
                            if not edges.empty:
                                links = edges.groupby(['Player', 'NextPlayer']).size().reset_index(name='count')
                                links = links[links['count'] > 2]
                                _mc1, _mc2 = st.columns(2)
                                _mc1.metric("Players in Network", len(avg_pos))
                                _mc2.metric("Pass Connections (≥3)", len(links))
                                fig_l_net = _make_plotly_pitch("Passing Network (Structure)")
                                max_edge = max(links['count'].max(), 1)
                                for _, _row in links.iterrows():
                                    p1, p2, c = _row['Player'], _row['NextPlayer'], _row['count']
                                    if p1 in avg_pos.index and p2 in avg_pos.index:
                                        fig_l_net.add_trace(go.Scatter(
                                            x=[avg_pos.loc[p1].x, avg_pos.loc[p2].x],
                                            y=[avg_pos.loc[p1].y, avg_pos.loc[p2].y],
                                            mode='lines',
                                            line=dict(color='#ff4b4b', width=1 + (4 * c / max_edge)),
                                            opacity=0.55, hoverinfo='skip', showlegend=False
                                        ))
                                node_sizes = np.clip(pass_counts.reindex(avg_pos.index).fillna(0).values * 2, 8, 40)
                                fig_l_net.add_trace(go.Scatter(
                                    x=avg_pos['x'], y=avg_pos['y'],
                                    mode='markers+text',
                                    marker=dict(size=node_sizes, color='#0e1117', line=dict(color='white', width=2)),
                                    text=[p.split(' ')[-1] for p in avg_pos.index],
                                    textposition='middle center', textfont=dict(color='white', size=9),
                                    customdata=np.column_stack([avg_pos.index.values, pass_counts.reindex(avg_pos.index).fillna(0).astype(int).values]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Events: %{customdata[1]}<extra></extra>",
                                    name='Players'
                                ))
                                st.plotly_chart(fig_l_net, use_container_width=True)
                            else:
                                st.info("No passing-network values available for current filters.")

                        # --- Out of Possession ---
                        if "Defensive Actions Map" in modules:
                            st.subheader("🔴 Defensive Actions Map")
                            def_map = viz_df[viz_df['Type'].isin([4, 7, 8, 12])].copy()
                            if not def_map.empty:
                                def_map['Action'] = def_map['Type'].map({4: 'Foul', 7: 'Tackle', 8: 'Interception', 12: 'Clearance'})
                                _n_tkl = len(def_map[def_map['Action'] == 'Tackle'])
                                _n_int = len(def_map[def_map['Action'] == 'Interception'])
                                _n_clr = len(def_map[def_map['Action'] == 'Clearance'])
                                _n_foul = len(def_map[def_map['Action'] == 'Foul'])
                                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                                _mc1.metric("🔵 Tackles", _n_tkl)
                                _mc2.metric("🟠 Interceptions", _n_int)
                                _mc3.metric("⬜ Clearances", _n_clr)
                                _mc4.metric("🔴 Fouls", _n_foul)
                                fig_dm = _make_plotly_pitch("Defensive Actions")
                                for action, color, symbol in [
                                    ('Tackle', '#00a3ff', 'diamond'),
                                    ('Interception', '#ff9900', 'square'),
                                    ('Clearance', '#d1d5db', 'triangle-up'),
                                    ('Foul', '#ff4b4b', 'x')
                                ]:
                                    adf = def_map[def_map['Action'] == action]
                                    if adf.empty:
                                        continue
                                    fig_dm.add_trace(go.Scatter(
                                        x=adf['x'], y=adf['y'], mode='markers', name=action,
                                        marker=dict(color=color, size=10, symbol=symbol, line=dict(color='white', width=1)),
                                        customdata=np.column_stack([adf['Player'], adf['Minute'], adf['Outcome']]),
                                        hovertemplate="<b>%{customdata[0]}</b><br>Minute: %{customdata[1]}'<br>Outcome: %{customdata[2]}<extra></extra>"
                                    ))
                                st.plotly_chart(fig_dm, use_container_width=True)
                                # --- Per-player defensive breakdown ---
                                _succ = def_map[def_map['Outcome'] == 'Successful']
                                _total_succ = len(_succ)
                                _total_acts = len(def_map)
                                _succ_rate = round(_total_succ / _total_acts * 100, 1) if _total_acts > 0 else 0
                                _tkl_won = len(def_map[(def_map['Action'] == 'Tackle') & (def_map['Outcome'] == 'Successful')])
                                _mc5, _mc6 = st.columns(2)
                                _mc5.metric("✅ Success Rate", f"{_succ_rate}%")
                                _mc6.metric("🔵 Tackles Won", _tkl_won)
                                with st.expander("👥 Player Defensive Breakdown"):
                                    # Build per-player pivot: action counts + success rate
                                    _pvt = def_map.groupby(['Player', 'Action']).size().unstack(fill_value=0).reset_index()
                                    for _col in ['Tackle', 'Interception', 'Clearance', 'Foul']:
                                        if _col not in _pvt.columns:
                                            _pvt[_col] = 0
                                    _pvt['Total'] = _pvt[['Tackle', 'Interception', 'Clearance', 'Foul']].sum(axis=1)
                                    _succ_by_player = def_map[def_map['Outcome'] == 'Successful'].groupby('Player').size().reset_index(name='Successful')
                                    _tkl_won_by_player = def_map[(def_map['Action'] == 'Tackle') & (def_map['Outcome'] == 'Successful')].groupby('Player').size().reset_index(name='Tackles Won')
                                    _pvt = _pvt.merge(_succ_by_player, on='Player', how='left').fillna(0)
                                    _pvt = _pvt.merge(_tkl_won_by_player, on='Player', how='left').fillna(0)
                                    _pvt['Success Rate %'] = (_pvt['Successful'] / _pvt['Total'] * 100).round(1)
                                    _pvt = _pvt.sort_values('Total', ascending=False)
                                    # Grouped bar chart
                                    _bar_data = []
                                    for _act, _col in [('Tackle', '#00a3ff'), ('Interception', '#ff9900'), ('Clearance', '#d1d5db'), ('Foul', '#ff4b4b')]:
                                        if _act in _pvt.columns:
                                            _bar_data.append(go.Bar(
                                                name=_act, y=_pvt['Player'], x=_pvt[_act],
                                                orientation='h', marker_color=_col
                                            ))
                                    fig_def_break = go.Figure(data=_bar_data)
                                    fig_def_break.update_layout(
                                        barmode='stack', template='plotly_dark',
                                        title='Defensive Actions by Player',
                                        xaxis_title='Count', yaxis_title='',
                                        height=max(260, len(_pvt) * 32),
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5)
                                    )
                                    st.plotly_chart(fig_def_break, use_container_width=True)
                                    # Success rate bar
                                    fig_sr = px.bar(
                                        _pvt.sort_values('Success Rate %'), x='Success Rate %', y='Player',
                                        orientation='h', template='plotly_dark',
                                        title='Defensive Success Rate by Player (%)',
                                        color='Success Rate %', color_continuous_scale='RdYlGn',
                                        range_color=[0, 100]
                                    )
                                    fig_sr.update_layout(height=max(240, len(_pvt) * 32), margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
                                    st.plotly_chart(fig_sr, use_container_width=True)
                                    # Summary table
                                    _display_cols = ['Player', 'Tackle', 'Tackles Won', 'Interception', 'Clearance', 'Foul', 'Total', 'Success Rate %']
                                    _display_cols = [c for c in _display_cols if c in _pvt.columns]
                                    st.dataframe(
                                        _pvt[_display_cols].rename(columns={'Tackle': 'Tackles', 'Interception': 'Interceptions', 'Clearance': 'Clearances'}).reset_index(drop=True),
                                        use_container_width=True, hide_index=True
                                    )
                            else:
                                st.info("No defensive actions recorded in this range.")

                        if "Defensive Shield (Heatmap + Line)" in modules:
                            st.subheader("🔴 Defensive Shield")
                            def_heat = viz_df[viz_df['Type'].isin([4, 7, 8, 12])]
                            if not def_heat.empty:
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("Total Defensive Actions", len(def_heat))
                                _mc2.metric("Players", def_heat['Player'].nunique())
                                _mc3.metric("🟡 Avg Recovery Height", f"{avg_rec_height} u", help="Dashed yellow line on the map")
                                fig_ds = _make_plotly_pitch("Defensive Density — brighter = more actions · 🟡 line = avg recovery height")
                                fig_ds.add_trace(go.Histogram2d(
                                    x=def_heat['x'], y=def_heat['y'], nbinsx=20, nbinsy=14,
                                    colorscale='Reds', opacity=0.7, showscale=True,
                                    colorbar=dict(title='Def Actions')
                                ))
                                fig_ds.add_vline(x=avg_rec_height, line_dash='dash', line_color='#ffd700', line_width=3)
                                st.plotly_chart(fig_ds, use_container_width=True)
                                def_shield_leaders = def_heat.groupby('Player').size().reset_index(name='Def Actions').sort_values('Def Actions', ascending=False)
                                if not def_shield_leaders.empty:
                                    with st.expander("👥 Defensive Leaders"):
                                        fig_dsl = px.bar(def_shield_leaders.head(8).sort_values('Def Actions'), x='Def Actions', y='Player', orientation='h', template='plotly_dark')
                                        fig_dsl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_dsl, use_container_width=True)
                            else:
                                st.info("No defensive actions to render.")

                        # --- Attacking Phase ---
                        if "Actions Leading to Shots" in modules:
                            st.subheader("⚔️ Actions Leading to Shots")
                            shots_df = viz_df[viz_df['Type'].isin([13, 14, 15, 16])].copy()
                            if not shots_df.empty:
                                _n_goals = len(shots_df[(shots_df['Type'] == 16) & (shots_df['Outcome'] != 'Own Goal')])
                                _n_saved = len(shots_df[shots_df['Type'] == 15])
                                _n_blocked = len(shots_df[shots_df['isBlocked']])
                                _n_missed = len(shots_df) - _n_goals - _n_saved - _n_blocked
                                _tot_xg = round(shots_df['xG'].sum(), 2)
                                _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
                                _mc1.metric("⭐ Goals", _n_goals)
                                _mc2.metric("🟡 Saved", _n_saved)
                                _mc3.metric("⬜ Blocked", _n_blocked)
                                _mc4.metric("🔴 Missed", _n_missed)
                                _mc5.metric("Total xG", _tot_xg)
                                shots_df['ShotResult'] = np.select(
                                    [
                                        (shots_df['Type'] == 16) & (shots_df['Outcome'] != 'Own Goal'),
                                        shots_df['isBlocked'],
                                        shots_df['Type'] == 15,
                                    ],
                                    ['Goal', 'Blocked', 'Saved'],
                                    default='Missed'
                                )
                                fig_l_shots = _make_plotly_pitch("Shot Outcomes")
                                style = {
                                    'Goal': ('#00ff85', 'star', 15),
                                    'Blocked': ('#aaaaaa', 'x', 10),
                                    'Saved': ('#ffd700', 'circle', 10),
                                    'Missed': ('#ff4b4b', 'diamond-open', 10),
                                }
                                for res in ['Goal', 'Saved', 'Blocked', 'Missed']:
                                    sdf = shots_df[shots_df['ShotResult'] == res]
                                    if sdf.empty:
                                        continue
                                    color, symbol, size = style[res]
                                    fig_l_shots.add_trace(go.Scatter(
                                        x=sdf['x'], y=sdf['y'], mode='markers', name=res,
                                        marker=dict(color=color, symbol=symbol, size=size, line=dict(color='white', width=1)),
                                        customdata=np.column_stack([sdf['Player'], sdf['Minute'], sdf['xG'].round(3)]),
                                        hovertemplate="<b>%{customdata[0]}</b><br>Minute: %{customdata[1]}'<br>xG: %{customdata[2]}<extra></extra>"
                                    ))
                                st.plotly_chart(fig_l_shots, use_container_width=True)

                                # --- Pre-shot action chain ---
                                _action_type_map = {1: 'Pass', 3: 'Dribble / Carry', 7: 'Tackle Won', 8: 'Interception'}
                                _pre_shot_rows = []
                                for _, _shot in shots_df.iterrows():
                                    _prev = viz_df[
                                        (viz_df['Index'] < _shot['Index']) &
                                        (viz_df['Index'] >= _shot['Index'] - 8) &
                                        (viz_df['Type'].isin([1, 3, 7, 8]))
                                    ].sort_values('Index')
                                    if not _prev.empty:
                                        _act = _prev.iloc[-1].copy()
                                        _act['ActionLabel'] = _action_type_map.get(_act['Type'], 'Other')
                                        _act['ShotResult'] = _shot['ShotResult']
                                        _act['ShotMinute'] = _shot['Minute']
                                        _pre_shot_rows.append(_act)

                                if _pre_shot_rows:
                                    _pre_df = pd.DataFrame(_pre_shot_rows)
                                    _n_pass_led = len(_pre_df[_pre_df['ActionLabel'] == 'Pass'])
                                    _n_drb_led = len(_pre_df[_pre_df['ActionLabel'] == 'Dribble / Carry'])
                                    _n_int_led = len(_pre_df[_pre_df['ActionLabel'] == 'Interception'])
                                    _n_tkl_led = len(_pre_df[_pre_df['ActionLabel'] == 'Tackle Won'])
                                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                                    _mc1.metric("🔵 Pass-Led", _n_pass_led)
                                    _mc2.metric("🟣 Dribble-Led", _n_drb_led)
                                    _mc3.metric("🟠 Interception-Led", _n_int_led)
                                    _mc4.metric("🔵 Tackle-Led", _n_tkl_led)

                                    _action_styles = {
                                        'Pass':             ('#00ffff', 'solid',  2, True),
                                        'Dribble / Carry':  ('#a855f7', 'solid',  2, False),
                                        'Interception':     ('#ff9900', 'dot',    2, False),
                                        'Tackle Won':       ('#00a3ff', 'dot',    2, False),
                                    }
                                    fig_pre = _make_plotly_pitch("Actions That Led to Shots — 🔵 Pass · 🟣 Dribble · 🟠 Interception · 🔷 Tackle")
                                    for _lbl, (_col, _dash, _w, _arr) in _action_styles.items():
                                        _sub = _pre_df[_pre_df['ActionLabel'] == _lbl]
                                        if not _sub.empty:
                                            _add_plotly_action_lines(fig_pre, _sub, _lbl, _col, dash=_dash, width=_w, arrows=_arr)
                                    st.plotly_chart(fig_pre, use_container_width=True)

                                    with st.expander("📊 Pre-Shot Action Breakdown"):
                                        _bc1, _bc2 = st.columns(2)
                                        with _bc1:
                                            st.markdown("**Actions by Type**")
                                            _type_counts = _pre_df['ActionLabel'].value_counts().reset_index()
                                            _type_counts.columns = ['Action Type', 'Count']
                                            _color_map = {'Pass': '#00ffff', 'Dribble / Carry': '#a855f7', 'Interception': '#ff9900', 'Tackle Won': '#00a3ff'}
                                            fig_tc = px.bar(_type_counts.sort_values('Count'), x='Count', y='Action Type',
                                                            orientation='h', template='plotly_dark',
                                                            color='Action Type', color_discrete_map=_color_map)
                                            fig_tc.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                                            st.plotly_chart(fig_tc, use_container_width=True)
                                        with _bc2:
                                            st.markdown("**Players — Pre-Shot Actions**")
                                            _player_counts = _pre_df.groupby(['Player', 'ActionLabel']).size().reset_index(name='Count')
                                            _player_total = _player_counts.groupby('Player')['Count'].sum().reset_index().sort_values('Count', ascending=False).head(8)
                                            _player_counts_top = _player_counts[_player_counts['Player'].isin(_player_total['Player'])]
                                            fig_pp = px.bar(_player_counts_top, x='Count', y='Player', color='ActionLabel',
                                                            orientation='h', template='plotly_dark',
                                                            color_discrete_map=_color_map, barmode='stack')
                                            fig_pp.update_layout(height=max(220, len(_player_total) * 30),
                                                                 margin=dict(l=0, r=0, t=10, b=0),
                                                                 legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5))
                                            st.plotly_chart(fig_pp, use_container_width=True)

                                xg_leaders = shots_df.groupby('Player')['xG'].sum().reset_index(name='Total xG').sort_values('Total xG', ascending=False)
                                shot_counts = shots_df.groupby('Player').size().reset_index(name='Shots')
                                xg_leaders = xg_leaders.merge(shot_counts, on='Player')
                                if not xg_leaders.empty:
                                    with st.expander("👥 Shot Leaders (xG)"):
                                        fig_xgl = px.bar(xg_leaders.head(8).sort_values('Total xG'), x='Total xG', y='Player', orientation='h', template='plotly_dark', hover_data=['Shots'])
                                        fig_xgl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_xgl, use_container_width=True)
                            else:
                                st.info("No shot actions available for current filters.")

                        if "Zone 14 & Half-Spaces" in modules:
                            st.subheader("⚔️ Zone 14 and Half-Spaces")

                            # Zone boundaries: (label, xmin, xmax, ymin, ymax, colour)
                            _zone_defs = [
                                ('Zone 14',          65, 85, 37, 63, '#ffd700'),
                                ('Left Half-Space',  65, 85, 20, 37, '#ff4b4b'),
                                ('Right Half-Space', 65, 85, 63, 80, '#ff4b4b'),
                            ]

                            def _in_any_zone(df, xc, yc):
                                mask = pd.Series(False, index=df.index)
                                for _, xmin, xmax, ymin, ymax, _ in _zone_defs:
                                    mask |= df[xc].between(xmin, xmax) & df[yc].between(ymin, ymax)
                                return df[mask].copy()

                            def _assign_zone(df, xc, yc):
                                def _lbl(r):
                                    for lbl, xmin, xmax, ymin, ymax, _ in _zone_defs:
                                        if xmin <= r[xc] <= xmax and ymin <= r[yc] <= ymax:
                                            return lbl
                                    return 'Other'
                                df['Zone'] = df.apply(_lbl, axis=1)
                                return df

                            # Passes whose endpoint lands in a zone
                            _z_passes = _in_any_zone(viz_df[viz_df['Type'] == 1].copy(), 'endX', 'endY')
                            _z_passes['Outcome'] = _z_passes['Outcome'].fillna('Unsuccessful')
                            _z_passes = _assign_zone(_z_passes, 'endX', 'endY')
                            _z_pass_succ = _z_passes[_z_passes['Outcome'] == 'Successful']
                            _z_pass_fail = _z_passes[_z_passes['Outcome'] != 'Successful']

                            # Dribbles / carries whose endpoint lands in a zone
                            _z_carries = _in_any_zone(viz_df[viz_df['Type'] == 3].copy(), 'endX', 'endY')
                            _z_carries['Outcome'] = _z_carries['Outcome'].fillna('Successful')
                            _z_carries = _assign_zone(_z_carries, 'endX', 'endY')
                            _z_carry_succ = _z_carries[_z_carries['Outcome'] == 'Successful']
                            _z_carry_fail = _z_carries[_z_carries['Outcome'] != 'Successful']

                            # Shots taken from within the zones (use start position)
                            _z_shots = _in_any_zone(viz_df[viz_df['Type'].isin([13, 14, 15, 16])].copy(), 'x', 'y')
                            _z_shots['Outcome'] = _z_shots['Outcome'].fillna('Unknown')
                            _z_shots = _assign_zone(_z_shots, 'x', 'y')

                            # Metrics row
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("📬 Passes into Zones", len(_z_passes))
                            _mc2.metric("✅ Successful Passes", len(_z_pass_succ))
                            _mc3.metric("🟣 Dribbles into Zones", len(_z_carries))
                            _mc4.metric("🎯 Shots from Zones", len(_z_shots))

                            # Pitch map
                            fig_l_zones = _make_plotly_pitch("Zone 14 🟡 · Left Half-Space 🔴 · Right Half-Space 🔴")
                            fig_l_zones.update_layout(
                                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                                            bgcolor='rgba(14,17,23,0.7)', bordercolor='#444', borderwidth=1),
                                margin=dict(l=10, r=10, t=45, b=60),
                            )
                            for _zlbl, xmin, xmax, ymin, ymax, zcolor in _zone_defs:
                                fig_l_zones.add_shape(
                                    type='rect', x0=xmin, y0=ymin, x1=xmax, y1=ymax,
                                    line=dict(color='white', width=1), fillcolor=zcolor, opacity=0.15
                                )
                            if not _z_pass_succ.empty:
                                _add_plotly_action_lines(fig_l_zones, _z_pass_succ, "✅ Pass (Successful)", "#00ff85", width=2, arrows=True)
                            if not _z_pass_fail.empty:
                                _add_plotly_action_lines(fig_l_zones, _z_pass_fail, "❌ Pass (Unsuccessful)", "#ff4b4b", width=2, dash='dot', arrows=True)
                            if not _z_carry_succ.empty:
                                _add_plotly_action_lines(fig_l_zones, _z_carry_succ, "🟣 Dribble (Successful)", "#a855f7", width=2)
                            if not _z_carry_fail.empty:
                                _add_plotly_action_lines(fig_l_zones, _z_carry_fail, "⚠️ Dribble (Unsuccessful)", "#ff9900", width=2, dash='dot')
                            if not _z_shots.empty:
                                _shot_type_sym = {16: 'star', 13: 'circle', 14: 'square', 15: 'x'}
                                _z_shot_sym = _z_shots['Type'].map(_shot_type_sym).fillna('circle').tolist()
                                _z_shot_custom = np.column_stack([
                                    _z_shots['Player'].astype(str),
                                    _z_shots['Minute'].astype(int),
                                    _z_shots['Outcome'].astype(str),
                                    _z_shots['Zone'].astype(str),
                                ])
                                fig_l_zones.add_trace(go.Scatter(
                                    x=_z_shots['x'], y=_z_shots['y'], mode='markers',
                                    marker=dict(color='#ff6b6b', size=10, symbol=_z_shot_sym,
                                                line=dict(color='white', width=1)),
                                    name='💥 Shot',
                                    customdata=_z_shot_custom,
                                    hovertemplate=(
                                        "<b>%{customdata[0]}</b><br>"
                                        "Minute: %{customdata[1]}'<br>"
                                        "Outcome: %{customdata[2]}<br>"
                                        "Zone: %{customdata[3]}<extra></extra>"
                                    ),
                                ))
                            st.plotly_chart(fig_l_zones, use_container_width=True)

                            # Player zone activity breakdown
                            _z_all_players = set()
                            for _zdf in [_z_passes, _z_carries, _z_shots]:
                                if not _zdf.empty:
                                    _z_all_players.update(_zdf['Player'].unique())

                            if _z_all_players:
                                with st.expander("👥 Player Zone Activity"):
                                    _z_rows = []
                                    for _pl in sorted(_z_all_players):
                                        _ps = len(_z_pass_succ[_z_pass_succ['Player'] == _pl]) if not _z_pass_succ.empty else 0
                                        _pf = len(_z_pass_fail[_z_pass_fail['Player'] == _pl]) if not _z_pass_fail.empty else 0
                                        _ds = len(_z_carry_succ[_z_carry_succ['Player'] == _pl]) if not _z_carry_succ.empty else 0
                                        _df2 = len(_z_carry_fail[_z_carry_fail['Player'] == _pl]) if not _z_carry_fail.empty else 0
                                        _sh = len(_z_shots[_z_shots['Player'] == _pl]) if not _z_shots.empty else 0
                                        _z_rows.append({
                                            'Player':         _pl,
                                            'Pass ✅':        _ps,
                                            'Pass ❌':        _pf,
                                            'Dribble ✅':     _ds,
                                            'Dribble ❌':     _df2,
                                            'Shots':          _sh,
                                            'Total':          _ps + _pf + _ds + _df2 + _sh,
                                        })
                                    _z_player_df = pd.DataFrame(_z_rows).sort_values('Total', ascending=False)
                                    _zpz1, _zpz2 = st.columns(2)
                                    with _zpz1:
                                        _z_bar_df = _z_player_df.melt(
                                            id_vars=['Player'],
                                            value_vars=['Pass ✅', 'Pass ❌', 'Dribble ✅', 'Dribble ❌', 'Shots'],
                                            var_name='Action', value_name='Count'
                                        )
                                        _z_bar_df = _z_bar_df[_z_bar_df['Count'] > 0]
                                        _z_color_map = {
                                            'Pass ✅':    '#00ff85',
                                            'Pass ❌':    '#ff4b4b',
                                            'Dribble ✅': '#a855f7',
                                            'Dribble ❌': '#ff9900',
                                            'Shots':      '#ff6b6b',
                                        }
                                        fig_z_bar = px.bar(
                                            _z_bar_df, x='Count', y='Player', color='Action',
                                            orientation='h', color_discrete_map=_z_color_map,
                                            template='plotly_dark', title='Zone Actions by Player',
                                            barmode='stack'
                                        )
                                        fig_z_bar.update_layout(
                                            height=max(260, len(_z_all_players) * 28),
                                            margin=dict(l=0, r=0, t=30, b=0)
                                        )
                                        st.plotly_chart(fig_z_bar, use_container_width=True)
                                    with _zpz2:
                                        st.dataframe(
                                            _z_player_df.reset_index(drop=True),
                                            use_container_width=True, hide_index=True
                                        )

                        if "Zone Invasions" in modules:
                            st.subheader("⚔️ Zone Invasions")

                            # --- Final Third: any action whose endpoint crosses into x > 66.7 ---
                            def _ft_mask(df):
                                return (df['endX'] > 66.7) & (df['x'] <= 66.7)

                            def _box_mask(df):
                                return (df['endX'] > 83) & (df['endY'].between(21, 79)) & \
                                       ~((df['x'] > 83) & (df['y'].between(21, 79)))

                            # Passes
                            _inv_pass_all = viz_df[viz_df['Type'] == 1].copy()
                            _inv_pass_all['Outcome'] = _inv_pass_all['Outcome'].fillna('Unsuccessful')
                            _inv_ft_pass_s  = _inv_pass_all[_ft_mask(_inv_pass_all) & (_inv_pass_all['Outcome'] == 'Successful')]
                            _inv_ft_pass_f  = _inv_pass_all[_ft_mask(_inv_pass_all) & (_inv_pass_all['Outcome'] != 'Successful')]
                            _inv_box_pass_s = _inv_pass_all[_box_mask(_inv_pass_all) & (_inv_pass_all['Outcome'] == 'Successful')]
                            _inv_box_pass_f = _inv_pass_all[_box_mask(_inv_pass_all) & (_inv_pass_all['Outcome'] != 'Successful')]

                            # Dribbles / carries
                            _inv_carry_all = viz_df[viz_df['Type'] == 3].copy()
                            _inv_carry_all['Outcome'] = _inv_carry_all['Outcome'].fillna('Successful')
                            _inv_ft_carry_s  = _inv_carry_all[_ft_mask(_inv_carry_all) & (_inv_carry_all['Outcome'] == 'Successful')]
                            _inv_ft_carry_f  = _inv_carry_all[_ft_mask(_inv_carry_all) & (_inv_carry_all['Outcome'] != 'Successful')]
                            _inv_box_carry_s = _inv_carry_all[_box_mask(_inv_carry_all) & (_inv_carry_all['Outcome'] == 'Successful')]
                            _inv_box_carry_f = _inv_carry_all[_box_mask(_inv_carry_all) & (_inv_carry_all['Outcome'] != 'Successful')]

                            # Shots taken from Final Third or Box
                            _inv_shots = viz_df[viz_df['Type'].isin([13, 14, 15, 16])].copy()
                            _inv_shots['Outcome'] = _inv_shots['Outcome'].fillna('Unknown')
                            _inv_shots_ft  = _inv_shots[_inv_shots['x'] > 66.7]
                            _inv_shots_box = _inv_shots[_inv_shots['x'] > 83 & _inv_shots['y'].between(21, 79)]
                            _inv_shots_all = pd.concat([_inv_shots_ft, _inv_shots_box]).drop_duplicates(subset=['Index']) if not (_inv_shots_ft.empty and _inv_shots_box.empty) else pd.DataFrame()

                            # Combined action counts per zone for metrics
                            _ft_all_idx  = pd.concat([_inv_ft_pass_s,  _inv_ft_pass_f,  _inv_ft_carry_s,  _inv_ft_carry_f]).drop_duplicates(subset=['Index'])  if any(not d.empty for d in [_inv_ft_pass_s, _inv_ft_pass_f, _inv_ft_carry_s, _inv_ft_carry_f]) else pd.DataFrame()
                            _box_all_idx = pd.concat([_inv_box_pass_s, _inv_box_pass_f, _inv_box_carry_s, _inv_box_carry_f]).drop_duplicates(subset=['Index']) if any(not d.empty for d in [_inv_box_pass_s, _inv_box_pass_f, _inv_box_carry_s, _inv_box_carry_f]) else pd.DataFrame()

                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("🟡 Final Third Actions", len(_ft_all_idx))
                            _mc2.metric("🔴 Box Actions", len(_box_all_idx))
                            _mc3.metric("🎯 Shots in Zones", len(_inv_shots_all) if not _inv_shots_all.empty else 0)
                            _all_inv_players = set()
                            for _d in [_ft_all_idx, _box_all_idx]:
                                if not _d.empty:
                                    _all_inv_players.update(_d['Player'].unique())
                            _mc4.metric("👤 Players", len(_all_inv_players))

                            fig_zi = _make_plotly_pitch("Zone Invasions — 🟡 Final Third · 🔴 Box Entries")
                            fig_zi.update_layout(
                                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                                            bgcolor='rgba(14,17,23,0.7)', bordercolor='#444', borderwidth=1),
                                margin=dict(l=10, r=10, t=45, b=60),
                            )
                            fig_zi.add_shape(type='rect', x0=66.7, y0=0, x1=100, y1=100,
                                            line=dict(color='#ffd700', width=1.5, dash='dot'),
                                            fillcolor='rgba(255,215,0,0.05)')
                            fig_zi.add_shape(type='rect', x0=83, y0=21, x1=100, y1=79,
                                            line=dict(color='#ff4b4b', width=2),
                                            fillcolor='rgba(255,75,75,0.07)')

                            # Passes — arrows
                            if not _inv_ft_pass_s.empty:
                                _add_plotly_action_lines(fig_zi, _inv_ft_pass_s,  "✅ Pass → Final Third",  "#00ff85", width=2, arrows=True)
                            if not _inv_ft_pass_f.empty:
                                _add_plotly_action_lines(fig_zi, _inv_ft_pass_f,  "❌ Pass → Final Third",  "#ff4b4b", width=2, dash='dot', arrows=True)
                            if not _inv_box_pass_s.empty:
                                _add_plotly_action_lines(fig_zi, _inv_box_pass_s, "✅ Pass → Box",          "#ffd700", width=2, arrows=True)
                            if not _inv_box_pass_f.empty:
                                _add_plotly_action_lines(fig_zi, _inv_box_pass_f, "❌ Pass → Box",          "#ff9900", width=2, dash='dot', arrows=True)
                            # Dribbles — solid/dotted lines
                            if not _inv_ft_carry_s.empty:
                                _add_plotly_action_lines(fig_zi, _inv_ft_carry_s,  "🟣 Dribble → Final Third ✅", "#a855f7", width=2)
                            if not _inv_ft_carry_f.empty:
                                _add_plotly_action_lines(fig_zi, _inv_ft_carry_f,  "⚠️ Dribble → Final Third ❌", "#c084fc", width=2, dash='dot')
                            if not _inv_box_carry_s.empty:
                                _add_plotly_action_lines(fig_zi, _inv_box_carry_s, "🟣 Dribble → Box ✅",         "#e879f9", width=2)
                            if not _inv_box_carry_f.empty:
                                _add_plotly_action_lines(fig_zi, _inv_box_carry_f, "⚠️ Dribble → Box ❌",         "#f0abfc", width=2, dash='dot')
                            # Shots
                            if not _inv_shots_all.empty:
                                _inv_shot_sym_map = {16: 'star', 13: 'circle', 14: 'square', 15: 'x'}
                                _inv_shot_syms = _inv_shots_all['Type'].map(_inv_shot_sym_map).fillna('circle').tolist()
                                _inv_shot_custom = np.column_stack([
                                    _inv_shots_all['Player'].astype(str),
                                    _inv_shots_all['Minute'].astype(int),
                                    _inv_shots_all['Outcome'].astype(str),
                                ])
                                fig_zi.add_trace(go.Scatter(
                                    x=_inv_shots_all['x'], y=_inv_shots_all['y'], mode='markers',
                                    marker=dict(color='#ff6b6b', size=10, symbol=_inv_shot_syms,
                                                line=dict(color='white', width=1)),
                                    name='💥 Shot',
                                    customdata=_inv_shot_custom,
                                    hovertemplate=(
                                        "<b>%{customdata[0]}</b><br>"
                                        "Minute: %{customdata[1]}'<br>"
                                        "Outcome: %{customdata[2]}<extra></extra>"
                                    ),
                                ))
                            st.plotly_chart(fig_zi, use_container_width=True)

                            # Player breakdown expander
                            if _all_inv_players:
                                with st.expander("👥 Player Zone Invasion Breakdown"):
                                    _inv_rows = []
                                    for _pl in sorted(_all_inv_players):
                                        def _cnt(df): return len(df[df['Player'] == _pl]) if not df.empty else 0
                                        _inv_rows.append({
                                            'Player':              _pl,
                                            'FT Pass ✅':          _cnt(_inv_ft_pass_s),
                                            'FT Pass ❌':          _cnt(_inv_ft_pass_f),
                                            'Box Pass ✅':         _cnt(_inv_box_pass_s),
                                            'Box Pass ❌':         _cnt(_inv_box_pass_f),
                                            'FT Dribble ✅':       _cnt(_inv_ft_carry_s),
                                            'FT Dribble ❌':       _cnt(_inv_ft_carry_f),
                                            'Box Dribble ✅':      _cnt(_inv_box_carry_s),
                                            'Box Dribble ❌':      _cnt(_inv_box_carry_f),
                                            'Shots':               _cnt(_inv_shots_all) if not _inv_shots_all.empty else 0,
                                        })
                                    _inv_player_df = pd.DataFrame(_inv_rows)
                                    _inv_player_df['Total'] = _inv_player_df.iloc[:, 1:].sum(axis=1)
                                    _inv_player_df = _inv_player_df.sort_values('Total', ascending=False)
                                    _inv_bar_df = _inv_player_df.melt(
                                        id_vars=['Player'],
                                        value_vars=['FT Pass ✅', 'FT Pass ❌', 'Box Pass ✅', 'Box Pass ❌',
                                                    'FT Dribble ✅', 'FT Dribble ❌', 'Box Dribble ✅', 'Box Dribble ❌', 'Shots'],
                                        var_name='Action', value_name='Count'
                                    )
                                    _inv_bar_df = _inv_bar_df[_inv_bar_df['Count'] > 0]
                                    _inv_color_map = {
                                        'FT Pass ✅':     '#00ff85', 'FT Pass ❌':     '#ff4b4b',
                                        'Box Pass ✅':    '#ffd700', 'Box Pass ❌':    '#ff9900',
                                        'FT Dribble ✅':  '#a855f7', 'FT Dribble ❌':  '#c084fc',
                                        'Box Dribble ✅': '#e879f9', 'Box Dribble ❌': '#f0abfc',
                                        'Shots':          '#ff6b6b',
                                    }
                                    _zpz1, _zpz2 = st.columns(2)
                                    with _zpz1:
                                        fig_zi_bar = px.bar(
                                            _inv_bar_df, x='Count', y='Player', color='Action',
                                            orientation='h', color_discrete_map=_inv_color_map,
                                            template='plotly_dark', title='Zone Invasion Actions by Player',
                                            barmode='stack'
                                        )
                                        fig_zi_bar.update_layout(
                                            height=max(260, len(_all_inv_players) * 28),
                                            margin=dict(l=0, r=0, t=30, b=0)
                                        )
                                        st.plotly_chart(fig_zi_bar, use_container_width=True)
                                    with _zpz2:
                                        st.dataframe(
                                            _inv_player_df.reset_index(drop=True),
                                            use_container_width=True, hide_index=True
                                        )

                        # --- Attacking Transition ---
                        if "Progression Trajectory Lines" in modules:
                            st.subheader("⚡ Progression Trajectory Lines")
                            ball_wins = plot_df[
                                (plot_df['Type'].isin([7, 8, 12])) &
                                ((plot_df['Outcome'] == 'Successful') | (plot_df['Type'].isin([8, 12])))
                            ].sort_values('Index')
                            _mc1, _mc2 = st.columns(2)
                            _mc1.metric("Ball Wins", len(ball_wins))
                            _mc2.metric("Players", ball_wins['Player'].nunique() if not ball_wins.empty else 0)
                            fig_ptl = _make_plotly_pitch("First 3–4 Actions After Ball Wins — each colour = one sequence")
                            palette = ['#00ff85', '#36d6e7', '#ffd700', '#ff7f50']
                            seq_count = 0
                            for _, win in ball_wins.head(24).iterrows():
                                seq = match_df[
                                    (match_df['Index'] > win['Index']) &
                                    (match_df['Index'] <= win['Index'] + 35) &
                                    (match_df['Team'] == sel_team) &
                                    (match_df['Type'].isin([1, 3]))
                                ].sort_values('Index').head(4)
                                if seq.empty:
                                    continue
                                color = palette[seq_count % len(palette)]
                                seq_count += 1
                                xs, ys = [], []
                                for _, r in seq.iterrows():
                                    xs.extend([r['x'], r['endX'], None])
                                    ys.extend([r['y'], r['endY'], None])
                                fig_ptl.add_trace(go.Scatter(
                                    x=xs, y=ys, mode='lines',
                                    line=dict(color=color, width=3),
                                    name='Transition Shape' if seq_count == 1 else 'Transition Shape',
                                    showlegend=(seq_count == 1),
                                    hoverinfo='skip'
                                ))
                                fig_ptl.add_trace(go.Scatter(
                                    x=seq['endX'], y=seq['endY'], mode='markers',
                                    marker=dict(color=color, size=8, line=dict(color='white', width=1)),
                                    showlegend=False,
                                    customdata=np.column_stack([seq['Player'], seq['Minute']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Minute: %{customdata[1]}'<extra></extra>"
                                ))

                            if seq_count > 0:
                                st.plotly_chart(fig_ptl, use_container_width=True)
                            else:
                                st.info("No ball-win transition sequences available in this range.")
                            bw_leaders = ball_wins.groupby('Player').size().reset_index(name='Ball Wins').sort_values('Ball Wins', ascending=False)
                            if not bw_leaders.empty:
                                with st.expander("👥 Ball Win Leaders"):
                                    fig_bwl = px.bar(bw_leaders.head(8).sort_values('Ball Wins'), x='Ball Wins', y='Player', orientation='h', template='plotly_dark')
                                    fig_bwl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                    st.plotly_chart(fig_bwl, use_container_width=True)

                        if "Time-to-Shot Scatter Plot" in modules:
                            st.subheader("⚡ Time-to-Shot Scatter Plot")
                            shots = plot_df[(plot_df['Type'].isin([13, 14, 15, 16])) & (plot_df['Outcome'] != 'Own Goal')].sort_values('Index')
                            _mc1, _mc2 = st.columns(2)
                            _mc1.metric("Shots Analysed", len(shots))
                            _mc2.metric("Goals", len(shots[shots['Type'] == 16]))
                            rows = []
                            for _, shot in shots.iterrows():
                                prior_opp = match_df[(match_df['Index'] < shot['Index']) & (match_df['Team'] != sel_team)].sort_values('Index').tail(1)
                                start_idx = int(prior_opp.iloc[0]['Index'] + 1) if not prior_opp.empty else int(shot['Index'])
                                chain = match_df[
                                    (match_df['Index'] >= start_idx) &
                                    (match_df['Index'] <= shot['Index']) &
                                    (match_df['Team'] == sel_team)
                                ].sort_values('Index')
                                if chain.empty:
                                    continue
                                first_evt = chain.iloc[0]
                                duration = float(shot['Minute'] - first_evt['Minute'])
                                if duration <= 0:
                                    duration = max(0.1, (shot['Index'] - first_evt['Index']) / 15)
                                rows.append({
                                    'Duration': duration,
                                    'xG': float(shot['xG']),
                                    'Actions': int(len(chain[chain['Type'].isin([1, 3])])),
                                    'Shooter': shot['Player'],
                                    'Minute': int(shot['Minute']),
                                    'Result': 'Goal' if shot['Type'] == 16 else 'No Goal'
                                })

                            tts_df = pd.DataFrame(rows)
                            if not tts_df.empty:
                                fig_tts = px.scatter(
                                    tts_df, x='Duration', y='xG', size='Actions', color='Result',
                                    color_discrete_map={'Goal': '#00ff85', 'No Goal': '#ff4b4b'},
                                    hover_data=['Shooter', 'Minute', 'Actions'],
                                    template='plotly_dark',
                                    title='Possession Duration vs Shot Quality'
                                )
                                fig_tts.update_xaxes(title='Possession Duration (minutes)')
                                fig_tts.update_yaxes(title='Resulting Shot xG')
                                st.plotly_chart(fig_tts, use_container_width=True)
                                tts_leaders = tts_df.groupby('Shooter')['xG'].sum().reset_index(name='Total xG').sort_values('Total xG', ascending=False)
                                if not tts_leaders.empty:
                                    with st.expander("👥 Shot Leaders"):
                                        fig_ttsl = px.bar(tts_leaders.head(8).sort_values('Total xG'), x='Total xG', y='Shooter', orientation='h', template='plotly_dark', labels={'Shooter': 'Player'})
                                        fig_ttsl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_ttsl, use_container_width=True)
                            else:
                                st.info("No shots to evaluate for transition speed.")

                        if "Transition Map" in modules:
                            st.subheader("⚡ Transition Map")
                            ball_wins_tm = plot_df[
                                (plot_df['Type'].isin([7, 8, 12])) &
                                ((plot_df['Outcome'] == 'Successful') | (plot_df['Type'].isin([8, 12])))
                            ].sort_values('Index')
                            _mc1, _mc2 = st.columns(2)
                            _mc1.metric("Ball Wins (transitions)", len(ball_wins_tm))
                            _mc2.metric("Players", ball_wins_tm['Player'].nunique() if not ball_wins_tm.empty else 0)
                            st.caption("🔵 Transition passes · 🟡 Shots · ⭐ Goals — hover for player, minute & xG")
                            fig_tm = _make_plotly_pitch("Transition Map — 🔵 Passes · 🟡 Shots · ⭐ Goals")
                            t_pass_xs, t_pass_ys = [], []
                            t_shot_rows, t_goal_rows, trans_event_frames = [], [], []
                            for _, win in ball_wins_tm.iterrows():
                                chain = match_df[
                                    (match_df['Index'] > win['Index']) &
                                    (match_df['Index'] <= win['Index'] + 40) &
                                    (match_df['Team'] == sel_team)
                                ].sort_values('Index')
                                opp_touch = match_df[
                                    (match_df['Index'] > win['Index']) &
                                    (match_df['Team'] != sel_team) &
                                    (match_df['Index'] <= win['Index'] + 40)
                                ].sort_values('Index')
                                if not opp_touch.empty:
                                    chain = chain[chain['Index'] < opp_touch.iloc[0]['Index']]
                                if chain.empty:
                                    continue
                                trans_event_frames.append(chain)
                                for _, p in chain[chain['Type'].isin([1, 3])].iterrows():
                                    t_pass_xs.extend([p['x'], p['endX'], None])
                                    t_pass_ys.extend([p['y'], p['endY'], None])
                                for _, s in chain[chain['Type'].isin([13, 14, 15])].iterrows():
                                    t_shot_rows.append({'x': s['x'], 'y': s['y'], 'Player': s['Player'], 'Minute': int(s['Minute']), 'xG': round(float(s['xG']), 3)})
                                for _, g in chain[(chain['Type'] == 16) & (chain['Outcome'] != 'Own Goal')].iterrows():
                                    t_goal_rows.append({'x': g['x'], 'y': g['y'], 'Player': g['Player'], 'Minute': int(g['Minute']), 'xG': round(float(g['xG']), 3)})
                            if t_pass_xs:
                                fig_tm.add_trace(go.Scatter(x=t_pass_xs, y=t_pass_ys, mode='lines', line=dict(color='#36d6e7', width=2), opacity=0.45, name='Transition Pass', hoverinfo='skip'))
                            if t_shot_rows:
                                sdf_tm = pd.DataFrame(t_shot_rows)
                                fig_tm.add_trace(go.Scatter(
                                    x=sdf_tm['x'], y=sdf_tm['y'], mode='markers', name='Shot',
                                    marker=dict(color='#ffd700', symbol='circle', size=12, line=dict(color='white', width=1)),
                                    customdata=np.column_stack([sdf_tm['Player'], sdf_tm['Minute'], sdf_tm['xG']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Shot @ %{customdata[1]}'<br>xG: %{customdata[2]}<extra></extra>"
                                ))
                            if t_goal_rows:
                                gdf_tm = pd.DataFrame(t_goal_rows)
                                fig_tm.add_trace(go.Scatter(
                                    x=gdf_tm['x'], y=gdf_tm['y'], mode='markers', name='Goal',
                                    marker=dict(color='#00ff85', symbol='star', size=17, line=dict(color='white', width=1.5)),
                                    customdata=np.column_stack([gdf_tm['Player'], gdf_tm['Minute'], gdf_tm['xG']]),
                                    hovertemplate="<b>%{customdata[0]}</b> ⚽<br>Goal @ %{customdata[1]}'<br>xG: %{customdata[2]}<extra></extra>"
                                ))
                            if t_pass_xs or t_shot_rows or t_goal_rows:
                                st.plotly_chart(fig_tm, use_container_width=True)
                                if trans_event_frames:
                                    trans_all_df = pd.concat(trans_event_frames)
                                    trans_leaders = trans_all_df.groupby('Player').size().reset_index(name='Transition Involvements').sort_values('Transition Involvements', ascending=False)
                                    if not trans_leaders.empty:
                                        with st.expander("👥 Transition Involvement Leaders"):
                                            fig_tml = px.bar(trans_leaders.head(8).sort_values('Transition Involvements'), x='Transition Involvements', y='Player', orientation='h', template='plotly_dark')
                                            fig_tml.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                            st.plotly_chart(fig_tml, use_container_width=True)
                            else:
                                st.info("No transition sequences found in this range.")

                        # --- Defensive Transition ---
                        if "Recovery vs. Loss Maps" in modules:
                            st.subheader("🧯 Recovery vs. Loss Maps")
                            if not transitions_df.empty:
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("🔴 Ball Losses", len(transitions_df))
                                _mc2.metric("🟢 Recoveries", len(transitions_df))
                                _avg_react = round((transitions_df['rec_idx'] - transitions_df['loss_idx']).mean(), 1) if not transitions_df.empty else 0
                                _mc3.metric("Avg Events to Recovery", _avg_react)
                                fig_rl = _make_plotly_pitch("Recovery vs Loss — 🔴 Loss location · 🟢 Recovery location")
                                xs, ys = [], []
                                for _, r in transitions_df.iterrows():
                                    xs.extend([r['loss_x'], r['rec_x'], None])
                                    ys.extend([r['loss_y'], r['rec_y'], None])
                                fig_rl.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#9ca3af', dash='dot', width=2), name='Recovery Link', hoverinfo='skip'))
                                fig_rl.add_trace(go.Scatter(
                                    x=transitions_df['loss_x'], y=transitions_df['loss_y'], mode='markers',
                                    marker=dict(color='#ff4b4b', size=10, symbol='x', line=dict(color='white', width=1)), name='Loss',
                                    customdata=np.column_stack([transitions_df['loss_player'], transitions_df['loss_min']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Loss Minute: %{customdata[1]}'<extra></extra>"
                                ))
                                fig_rl.add_trace(go.Scatter(
                                    x=transitions_df['rec_x'], y=transitions_df['rec_y'], mode='markers',
                                    marker=dict(color='#00ff85', size=10, symbol='circle', line=dict(color='white', width=1)), name='Recovery',
                                    customdata=np.column_stack([transitions_df['rec_player'], transitions_df['rec_min']]),
                                    hovertemplate="<b>%{customdata[0]}</b><br>Recovery Minute: %{customdata[1]}'<extra></extra>"
                                ))
                                st.plotly_chart(fig_rl, use_container_width=True)
                                rec_leaders = transitions_df.groupby('rec_player').size().reset_index(name='Recoveries').sort_values('Recoveries', ascending=False).rename(columns={'rec_player': 'Player'})
                                if not rec_leaders.empty:
                                    with st.expander("👥 Recovery Leaders"):
                                        fig_rcl = px.bar(rec_leaders.head(8).sort_values('Recoveries'), x='Recoveries', y='Player', orientation='h', template='plotly_dark')
                                        fig_rcl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_rcl, use_container_width=True)
                            else:
                                st.info("No loss-to-recovery pairs found in this range.")

                        if "Defensive Reaction Time/Distance Curves" in modules:
                            st.subheader("🧯 Defensive Reaction Time/Distance Curves")
                            if not transitions_df.empty:
                                st.caption("🔵 dots = individual transitions · 🟡 line = average curve — X axis: event count between loss and recovery")
                                transitions_df['Reaction Events'] = transitions_df['rec_idx'] - transitions_df['loss_idx']
                                transitions_df['Recovery Distance'] = np.sqrt(
                                    (transitions_df['rec_x'] - transitions_df['loss_x']) ** 2 +
                                    (transitions_df['rec_y'] - transitions_df['loss_y']) ** 2
                                )
                                curve = transitions_df.groupby('Reaction Events', as_index=False)['Recovery Distance'].mean().sort_values('Reaction Events')

                                fig_rt = go.Figure()
                                fig_rt.add_trace(go.Scatter(
                                    x=transitions_df['Reaction Events'], y=transitions_df['Recovery Distance'],
                                    mode='markers', marker=dict(color='#36d6e7', size=9),
                                    customdata=np.column_stack([transitions_df['loss_player'], transitions_df['rec_player']]),
                                    name='Transitions',
                                    hovertemplate="Loss: %{customdata[0]}<br>Recovery: %{customdata[1]}<br>Event Gap: %{x}<br>Distance: %{y:.1f}<extra></extra>"
                                ))
                                fig_rt.add_trace(go.Scatter(
                                    x=curve['Reaction Events'], y=curve['Recovery Distance'], mode='lines+markers',
                                    line=dict(color='#ffd700', width=3), marker=dict(size=7), name='Average Curve'
                                ))
                                fig_rt.update_layout(
                                    template='plotly_dark',
                                    title='Event-Based Defensive Reaction Profile',
                                    xaxis_title='Events Between Loss and Recovery (proxy)',
                                    yaxis_title='Pitch Distance Covered (Opta units)'
                                )
                                st.caption("Tracking sprint-speed is unavailable in event data, so this uses an event-gap and distance proxy.")
                                st.plotly_chart(fig_rt, use_container_width=True)
                            else:
                                st.info("Insufficient transition pairs to model defensive reaction curves.")

                        # --- Set Pieces ---
                        if "Set Piece Targeting (Corners)" in modules:
                            st.subheader("🎯 Set Piece Targeting (Corners)")
                            corners = viz_df[(viz_df['Type'] == 1) & (viz_df['isCorner'])]
                            if not corners.empty:
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("Total Corners", len(corners))
                                _mc2.metric("✅ Completed", len(corners[corners['Outcome'] == 'Successful']))
                                _mc3.metric("❌ Incomplete", len(corners[corners['Outcome'] == 'Unsuccessful']))
                                fig_corners = _make_plotly_pitch("Corner Delivery — ✅ Completed · ❌ Incomplete")
                                _add_plotly_action_lines(fig_corners, corners[corners['Outcome'] == 'Successful'], "✅ Completed Corner", "#00ffff", width=2, arrows=True)
                                _add_plotly_action_lines(fig_corners, corners[corners['Outcome'] == 'Unsuccessful'], "❌ Incomplete Corner", "#ff4b4b", width=2, dash='dot', arrows=True)
                                st.plotly_chart(fig_corners, use_container_width=True)
                                corner_leaders = corners.groupby('Player').size().reset_index(name='Corners').sort_values('Corners', ascending=False)
                                if not corner_leaders.empty:
                                    with st.expander("👥 Corner Delivery Leaders"):
                                        fig_cornl = px.bar(corner_leaders.head(8).sort_values('Corners'), x='Corners', y='Player', orientation='h', template='plotly_dark')
                                        fig_cornl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_cornl, use_container_width=True)
                            else:
                                st.info("No corners recorded in this range.")

                        if "Free Kick Targeting" in modules:
                            st.subheader("🎯 Free Kick Targeting")
                            free_kicks = viz_df[(viz_df['Type'] == 1) & (viz_df['isFreeKick'])]
                            if not free_kicks.empty:
                                _mc1, _mc2, _mc3 = st.columns(3)
                                _mc1.metric("Total Free Kicks", len(free_kicks))
                                _mc2.metric("✅ Completed", len(free_kicks[free_kicks['Outcome'] == 'Successful']))
                                _mc3.metric("Takers", free_kicks['Player'].nunique())
                                fig_fk = _make_plotly_pitch("Free-Kick Delivery — ✅ Completed · ❌ Incomplete")
                                _add_plotly_action_lines(fig_fk, free_kicks[free_kicks['Outcome'] == 'Successful'], "✅ Completed FK", "#00ffff", width=2, arrows=True)
                                _add_plotly_action_lines(fig_fk, free_kicks[free_kicks['Outcome'] == 'Unsuccessful'], "❌ Incomplete FK", "#ff4b4b", width=2, dash='dot', arrows=True)
                                st.plotly_chart(fig_fk, use_container_width=True)
                                fk_leaders = free_kicks.groupby('Player').size().reset_index(name='Free Kicks').sort_values('Free Kicks', ascending=False)
                                if not fk_leaders.empty:
                                    with st.expander("👥 Free Kick Leaders"):
                                        fig_fkl = px.bar(fk_leaders.head(8).sort_values('Free Kicks'), x='Free Kicks', y='Player', orientation='h', template='plotly_dark')
                                        fig_fkl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_fkl, use_container_width=True)
                            else:
                                st.info("No free kicks recorded in this range.")

                        # --- Goalkeeping ---
                        if "Shot Trajectory Map (GK View)" in modules:
                            st.subheader("🧤 Shot Trajectory Map (GK View)")
                            faced = opp_stats[(opp_stats['Type'].isin([13, 14, 15, 16])) & (opp_stats['Outcome'] != 'Own Goal')].copy()
                            if not faced.empty:
                                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                                _mc1.metric("Shots Faced", len(faced))
                                _mc2.metric("🔴 Goals Conceded", len(faced[faced['Type'] == 16]))
                                _mc3.metric("🔵 Saved/Missed", len(faced[faced['Type'] != 16]))
                                _mc4.metric("Opp xG vs", round(faced['xG'].sum(), 2))
                                st.caption("X axis: lateral offset from goal centre (left ← 0 → right) · Y axis: distance from goal line")
                                faced['DepthToGoal'] = 100 - faced['x']
                                faced['LateralFromCenter'] = faced['y'] - 50
                                faced['Result'] = np.where(faced['Type'] == 16, 'Goal Conceded', 'Saved/Missed')
                                fig_gk = px.scatter(
                                    faced,
                                    x='LateralFromCenter', y='DepthToGoal',
                                    color='Result',
                                    color_discrete_map={'Goal Conceded': '#ff4b4b', 'Saved/Missed': '#00a3ff'},
                                    hover_data=['Player', 'Minute', 'xG'],
                                    template='plotly_dark',
                                    title='Opponent Shot Origins from Goalkeeper Perspective'
                                )
                                fig_gk.update_xaxes(title='Lateral Offset from Goal Center (Left <-> Right)')
                                fig_gk.update_yaxes(title='Distance from Goal Line')
                                st.plotly_chart(fig_gk, use_container_width=True)
                                threat_leaders = faced.groupby('Player')['xG'].sum().reset_index(name='Total xG').sort_values('Total xG', ascending=False)
                                if not threat_leaders.empty:
                                    with st.expander("👥 Most Threatening Players (Opp.)"):
                                        fig_thl = px.bar(threat_leaders.head(8).sort_values('Total xG'), x='Total xG', y='Player', orientation='h', template='plotly_dark')
                                        fig_thl.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                                        st.plotly_chart(fig_thl, use_container_width=True)
                            else:
                                st.info("No opponent shots faced in this range.")

                        if "Goal Kick Direction Map" in modules:
                            st.subheader("🧤 Goal Kick Direction Map")
                            gkicks = viz_df[viz_df['Type'] == 52]
                            if not gkicks.empty:
                                _mc1, _mc2 = st.columns(2)
                                _mc1.metric("Goal Kicks", len(gkicks))
                                _mc2.metric("Takers", gkicks['Player'].nunique())
                                fig_gkd = _make_plotly_pitch("Goal Kick Direction and Target Zones")
                                _add_plotly_action_lines(fig_gkd, gkicks, "Goal Kick", "#3b82f6", width=3, arrows=True)
                                st.plotly_chart(fig_gkd, use_container_width=True)
                            else:
                                st.info("No goal kicks recorded in this range.")

                        st.divider()

                    else:
                        st.error(f"Error: {err}")

        # === TAB C: AVERAGE PLAYER STATS ===
        with sub_t3:
            mgr_short = manager.split()[-1].title()  # Amorim / Fletcher / Carrick
            st.header(f"👥 Average Player Stats")
            with st.spinner(f"Loading all {mgr_short}-era match data..."):
                utd_df, sp_goals_map, sp_assists_map, team_match_stats = _load_all_manager_data(manager)

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
                    tackles_all = pdf[pdf['Type'] == 7]
                    tackles_won = len(tackles_all[tackles_all['Outcome'] == 'Successful'])
                    tackles_lost = len(tackles_all[tackles_all['Outcome'] == 'Unsuccessful'])
                    tackles = len(tackles_all)
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
                        'TklW/Match': round(tackles_won / mp, 2),
                        'TklL/Match': round(tackles_lost / mp, 2),
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
                    edges = pd.DataFrame(columns=['Player', 'NextPlayer', 'count'])
                    if not connections.empty:
                        edges = connections.groupby(['Player', 'NextPlayer']).size().reset_index(name='count')
                        edges = edges[edges['count'] > min_edge_count]
                    # Players with at least one connection
                    connected_players = set(edges['Player'].tolist() + edges['NextPlayer'].tolist()) - {'Unknown'}
                    connected_players = sorted(connected_players)

                    selected_player_net = st.selectbox(
                        "Highlight a player's connections:",
                        options=["All Players"] + connected_players,
                        key=f"net_player_{mgr_short}"
                    )

                    fig_net, ax_net = plt.subplots(figsize=(10, 7))
                    fig_net.set_facecolor('#0e1117')
                    ax_net.set_facecolor('#0e1117')
                    pitch_net = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                    pitch_net.draw(ax=ax_net)
                    if not edges.empty:
                        max_edge = edges['count'].max()
                        if selected_player_net == "All Players":
                            draw_edges = edges
                        else:
                            draw_edges = edges[(edges['Player'] == selected_player_net) | (edges['NextPlayer'] == selected_player_net)]
                        for _, row in draw_edges.iterrows():
                            p1, p2 = row['Player'], row['NextPlayer']
                            if p1 in avg_pos.index and p2 in avg_pos.index:
                                width = (row['count'] / max_edge) * 6
                                alpha = min(0.9, row['count'] / max_edge + 0.2)
                                pitch_net.lines(avg_pos.loc[p1].x, avg_pos.loc[p1].y,
                                                avg_pos.loc[p2].x, avg_pos.loc[p2].y,
                                                lw=width, color='#ff4b4b', alpha=alpha, ax=ax_net, zorder=1)
                        # Only show connected players
                        if selected_player_net == "All Players":
                            show_players = connected_players
                        else:
                            show_players = set(draw_edges['Player'].tolist() + draw_edges['NextPlayer'].tolist()) - {'Unknown'}
                        for player in show_players:
                            if player in avg_pos.index and player in pass_counts.index:
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

        # === TAB D: AVERAGE TEAM STATS ===
        with sub_t4:
            mgr_short_t = manager.split()[-1].title()
            st.header(f"📈 Average Team Stats — {mgr_short_t} Era")
            with st.spinner(f"Loading all {mgr_short_t}-era match data..."):
                utd_df_t, _, _, team_stats_t = _load_all_manager_data(manager)

            if team_stats_t:
                ts_df = pd.DataFrame(team_stats_t)
                n_matches_t = len(ts_df)

                # --- Summary Metrics ---
                avg_poss = round(ts_df['Possession'].mean(), 1)
                avg_xg = round(ts_df['xG'].mean(), 2)
                avg_tilt = round(ts_df['Field Tilt'].mean(), 1)
                avg_ppda = round(ts_df['PPDA'].mean(), 1)
                avg_def = round(ts_df['Def Line'].mean(), 1)

                st.subheader(f"📊 Averages Across {n_matches_t} Matches")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("📊 Possession", f"{avg_poss}%")
                mc2.metric("📈 xG", f"{avg_xg}")
                mc3.metric("⚖️ Field Tilt", f"{avg_tilt}%")
                mc4.metric("🛑 PPDA", f"{avg_ppda}")
                mc5.metric("📏 Def. Line", f"{avg_def}m")
                st.divider()

                # --- Per-Match Table ---
                st.subheader("📋 Per-Match Breakdown")
                display_ts = ts_df.copy()
                display_ts.index = range(1, len(display_ts) + 1)
                st.dataframe(display_ts, use_container_width=True)
                st.divider()

                # --- Bar Charts Row ---
                bc1, bc2 = st.columns(2)
                with bc1:
                    fig_poss = px.bar(
                        ts_df, x='Match', y='Possession', title='Possession % per Match',
                        template='plotly_dark', color='Possession',
                        color_continuous_scale=['#ff4b4b', '#00ff85']
                    )
                    fig_poss.add_hline(y=avg_poss, line_dash='dash', line_color='#ffd700',
                                       annotation_text=f'Avg: {avg_poss}%', annotation_font_color='#ffd700')
                    fig_poss.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_poss, use_container_width=True)
                with bc2:
                    fig_xg = px.bar(
                        ts_df, x='Match', y='xG', title='xG per Match',
                        template='plotly_dark', color='xG',
                        color_continuous_scale=['#ff4b4b', '#00ff85']
                    )
                    fig_xg.add_hline(y=avg_xg, line_dash='dash', line_color='#ffd700',
                                     annotation_text=f'Avg: {avg_xg}', annotation_font_color='#ffd700')
                    fig_xg.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_xg, use_container_width=True)

                bc3, bc4 = st.columns(2)
                with bc3:
                    fig_tilt = px.bar(
                        ts_df, x='Match', y='Field Tilt', title='Field Tilt % per Match',
                        template='plotly_dark', color='Field Tilt',
                        color_continuous_scale=['#ff4b4b', '#00ff85']
                    )
                    fig_tilt.add_hline(y=avg_tilt, line_dash='dash', line_color='#ffd700',
                                       annotation_text=f'Avg: {avg_tilt}%', annotation_font_color='#ffd700')
                    fig_tilt.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_tilt, use_container_width=True)
                with bc4:
                    fig_ppda = px.bar(
                        ts_df, x='Match', y='PPDA', title='PPDA per Match',
                        template='plotly_dark', color='PPDA',
                        color_continuous_scale=['#00ff85', '#ff4b4b']
                    )
                    fig_ppda.add_hline(y=avg_ppda, line_dash='dash', line_color='#ffd700',
                                       annotation_text=f'Avg: {avg_ppda}', annotation_font_color='#ffd700')
                    fig_ppda.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_ppda, use_container_width=True)

                st.divider()

                # --- Average Team Pass Sonar ---
                if not utd_df_t.empty:
                    col_sonar, col_def = st.columns(2)
                    with col_sonar:
                        st.subheader("📡 Average Team Pass Sonar")
                        t_passes = utd_df_t[(utd_df_t['Type'] == 1) & (utd_df_t['Outcome'] == 'Successful')]
                        if not t_passes.empty:
                            dx = t_passes['endX'] - t_passes['x']
                            dy = t_passes['endY'] - t_passes['y']
                            angles = np.arctan2(dy, dx)
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
                            fig_ts.set_facecolor('#0e1117')
                            ax_ts.set_facecolor('#0e1117')
                            ax_ts.hist(angles, bins=24, color='#00ffff', alpha=0.7, edgecolor='white')
                            ax_ts.set_theta_zero_location('E')
                            ax_ts.set_yticks([])
                            ax_ts.grid(color='#262730')
                            ax_ts.tick_params(axis='x', colors='white')
                            ax_ts.set_title(f'Team Pass Direction — {n_matches_t} Matches', color='white', fontsize=12, pad=20)
                            st.pyplot(fig_ts)
                            plt.close(fig_ts)

                    # --- Average Defensive Heatmap + Line ---
                    with col_def:
                        st.subheader("🛡️ Average Defensive Heatmap")
                        def_events_t = utd_df_t[utd_df_t['Type'].isin([4, 7, 8, 12])]
                        fig_dh, ax_dh = plt.subplots(figsize=(10, 7))
                        fig_dh.set_facecolor('#0e1117')
                        ax_dh.set_facecolor('#0e1117')
                        pitch_dh = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                        pitch_dh.draw(ax=ax_dh)
                        if not def_events_t.empty and len(def_events_t) >= 2:
                            pitch_dh.kdeplot(def_events_t['x'].values, def_events_t['y'].values,
                                             ax=ax_dh, cmap='viridis', fill=True, levels=100, alpha=0.6)
                        if avg_def > 0:
                            pitch_dh.lines(avg_def, 0, avg_def, 100, color='#ffd700', lw=3,
                                           linestyle='dashed', alpha=0.9, ax=ax_dh,
                                           label=f'Avg Def Line ({avg_def}m)')
                            ax_dh.add_patch(mpatches.Rectangle((0, 0), avg_def, 100, alpha=0.1, color='#ffd700', ec=None))
                            ax_dh.legend(facecolor='#262730', labelcolor='white')
                        ax_dh.set_title(f'Defensive Actions — {n_matches_t} Matches', color='white', fontsize=12)
                        st.pyplot(fig_dh)
                        plt.close(fig_dh)

                    st.divider()

                    # --- Average Attacking Zones (5 Lanes) ---
                    col_az, col_xt = st.columns(2)
                    with col_az:
                        st.subheader("⚔️ Average Attacking Zones (5 Lanes)")
                        f3_acts_t = utd_df_t[(utd_df_t['x'] > 66.6) & (utd_df_t['Type'] == 1) & (utd_df_t['Outcome'] == 'Successful')]
                        total_f3_t = len(f3_acts_t)
                        fig_az, ax_az = plt.subplots(figsize=(10, 7))
                        fig_az.set_facecolor('#0e1117')
                        ax_az.set_facecolor('#0e1117')
                        pitch_az = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                        pitch_az.draw(ax=ax_az)
                        if total_f3_t > 0:
                            lane_defs = [
                                ("Right Flank", 0, 20),
                                ("Right Half", 20, 37),
                                ("Center", 37, 63),
                                ("Left Half", 63, 80),
                                ("Left Flank", 80, 100),
                            ]
                            for label, y_lo, y_hi in lane_defs:
                                count = len(f3_acts_t[(f3_acts_t['y'] > y_lo) & (f3_acts_t['y'] <= y_hi)]) if y_lo > 0 else len(f3_acts_t[f3_acts_t['y'] <= y_hi])
                                pct = (count / total_f3_t) * 100
                                height = y_hi - y_lo
                                rect = mpatches.Rectangle((66.6, y_lo), 33.4, height, alpha=min(0.9, max(0.2, pct / 40)), color='#ff4b4b', ec='white')
                                ax_az.add_patch(rect)
                                ax_az.text(66.6 + 16.7, y_lo + (height / 2), f"{label}\n{pct:.1f}%", color='white', ha='center', va='center', fontweight='bold', fontsize=9)
                        ax_az.set_title(f'Final 3rd Attacking Lanes — {n_matches_t} Matches', color='white', fontsize=12)
                        st.pyplot(fig_az)
                        plt.close(fig_az)

                    # --- Average xT Grid ---
                    with col_xt:
                        st.subheader("⚡ Average xT Grid")
                        xt_acts_t = utd_df_t[(utd_df_t['Type'].isin([1, 3])) & (utd_df_t['Outcome'] == 'Successful') & (utd_df_t['xT_Added'] > 0)]
                        fig_xtg, ax_xtg = plt.subplots(figsize=(10, 7))
                        fig_xtg.set_facecolor('#0e1117')
                        ax_xtg.set_facecolor('#0e1117')
                        pitch_xtg = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white')
                        pitch_xtg.draw(ax=ax_xtg)
                        if not xt_acts_t.empty:
                            bin_stat = pitch_xtg.bin_statistic(xt_acts_t['x'].values, xt_acts_t['y'].values, values=xt_acts_t['xT_Added'].values, statistic='sum', bins=(12, 8))
                            pitch_xtg.heatmap(bin_stat, ax=ax_xtg, cmap='magma', alpha=0.7, edgecolors='#262730', lw=1, zorder=0)
                            stat = bin_stat['statistic']
                            cx = bin_stat['cx']
                            cy = bin_stat['cy']
                            for row_i in range(stat.shape[0]):
                                for col_j in range(stat.shape[1]):
                                    val = stat[row_i, col_j]
                                    if val > 0.001:
                                        ax_xtg.text(cx[row_i, col_j], cy[row_i, col_j], f'{val:.3f}', color='white',
                                                    ha='center', va='center', fontsize=9, fontweight='bold', zorder=1)
                        else:
                            pitch_xtg.annotate("No positive xT actions recorded", xy=(50, 50), c='white', ha='center', va='center', size=15, ax=ax_xtg)
                        ax_xtg.set_title(f'Expected Threat Grid — {n_matches_t} Matches', color='white', fontsize=12)
                        st.pyplot(fig_xtg)
                        plt.close(fig_xtg)
            else:
                st.error(f"Failed to load {mgr_short_t} match data.")
