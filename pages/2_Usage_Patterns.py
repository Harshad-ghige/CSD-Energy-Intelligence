"""
CSD Energy Intelligence - Module 3: Usage Patterns
Individual Data Science & Big Data (DSBD) Laboratory Project
Department of Computer Science & Design (CSD)
K. K. Wagh Institute of Engineering Education & Research, Nashik
"""

import base64
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import filter_dataset
from src.ui_components import (
    get_cached_dataset,
    inject_custom_css,
    render_sidebar_department_info,
    render_sidebar_institutional_footer,
    render_sidebar_tariff_setting,
)

# Page configuration
st.set_page_config(
    page_title="Usage Patterns - CSD Energy Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject shared styling and department sidebar
inject_custom_css()
render_sidebar_department_info()
tariff_rate = render_sidebar_tariff_setting()
render_sidebar_institutional_footer()


# -----------------------------------------------------------------------------
# Asset Helpers (Cached Base64 for instant rendering)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def get_base64_image(file_path: str) -> str:
    """Reads and encodes local image asset to base64 data URI."""
    p = Path(file_path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    try:
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Module 3 Custom CSS
# -----------------------------------------------------------------------------
st.html(
    """
    <style>
    /* Replace entrypoint 'app' text with 'Overview' in sidebar nav */
    [data-testid="stSidebarNavItems"] li:first-child a span {
        display: none !important;
    }
    [data-testid="stSidebarNavItems"] li:first-child a::after {
        content: "Overview" !important;
        color: #A7D8BE !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
    }

    /* Institutional Header Card */
    .m3-header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0 0 0 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
        overflow: hidden;
        height: 125px;
        min-height: 125px;
        position: relative;
    }
    .m3-header-left {
        display: flex;
        align-items: center;
        gap: 16px;
        z-index: 2;
        padding: 12px 0;
        flex-shrink: 0;
    }
    .m3-icon-badge {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background: #E8F5E9;
        border: 1px solid #C8E6C9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        flex-shrink: 0;
    }
    .m3-header-title {
        font-size: 1.52rem;
        font-weight: 800;
        color: #0F3A2E;
        margin: 0;
        line-height: 1.18;
        letter-spacing: -0.01em;
    }
    .m3-header-subtitle {
        font-size: 0.88rem;
        font-weight: 600;
        color: #2D6A4F;
        margin-top: 3px;
    }
    .m3-header-meta {
        display: flex;
        align-items: center;
        gap: 18px;
        font-size: 0.77rem;
        color: #6B7280;
        margin-top: 5px;
        flex-wrap: wrap;
    }
    .m3-header-meta span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .m3-header-right {
        display: flex;
        align-items: center;
        position: relative;
        height: 100%;
    }
    .m3-building-img {
        height: 125px;
        width: auto;
        max-width: 440px;
        object-fit: cover;
        display: block;
    }

    /* DEMO MODE Notice Banner */
    .demo-amber-banner {
        background-color: #FEF9EE;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 8px 14px;
        color: #92400E;
        font-size: 0.82rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    /* Section Badges and Headings */
    .section-badge-circle {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background-color: #1E754A;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .sec-title-text {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0F3A2E;
        margin: 0;
        line-height: 1.25;
    }
    .sec-desc-text {
        font-size: 0.80rem;
        color: #6B7280;
        margin-top: 2px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        box-sizing: border-box;
    }
    .kpi-card-green {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
    }
    .kpi-card-blue {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
    }
    .kpi-card-purple {
        background-color: #FAF5FF;
        border: 1px solid #E9D5FF;
    }
    .kpi-card-peach {
        background-color: #FFF7ED;
        border: 1px solid #FED7AA;
    }

    .kpi-icon-bubble {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.0rem;
        flex-shrink: 0;
    }
    .bubble-green { background: #DCFCE7; color: #15803D; }
    .bubble-blue { background: #DBEAFE; color: #1D4ED8; }
    .bubble-purple { background: #F3E8FF; color: #7E22CE; }
    .bubble-peach { background: #FFEDD5; color: #C2410C; }

    .kpi-content-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
        flex: 1;
    }
    .kpi-label-title {
        font-size: 0.70rem;
        font-weight: 600;
        color: #6B7280;
        line-height: 1.15;
        margin-bottom: 2px;
    }
    .kpi-primary-val {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0F3A2E;
        line-height: 1.15;
        white-space: nowrap;
    }
    .kpi-secondary-text {
        font-size: 0.72rem;
        color: #6B7280;
        white-space: nowrap;
    }

    /* Chart Container Card */
    .chart-panel-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.94rem;
        font-weight: 700;
        color: #0F3A2E;
    }

    /* Dynamic Observation Box */
    .observation-box {
        background: #F0FDF4;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.81rem;
        color: #065F46;
        line-height: 1.5;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-top: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    /* Configuration Panel Card */
    .config-assumption-card {
        background: #F8FAF9;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
    .config-badge-academic {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .config-badge-off {
        background: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* Interpretation Callout */
    .academic-note-box {
        background: #F8FAF9;
        border-left: 3px solid #1E754A;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.78rem;
        color: #374151;
        line-height: 1.5;
        margin-top: 10px;
    }

    /* Visual Context Cards */
    .context-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        height: 100%;
        display: flex;
        gap: 14px;
        align-items: center;
    }
    .context-img-box {
        width: 130px;
        height: 85px;
        border-radius: 6px;
        overflow: hidden;
        flex-shrink: 0;
        background: #F3F4F6;
    }
    .context-img-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    </style>
    """
)

# -----------------------------------------------------------------------------
# 1. Institutional Header Card
# -----------------------------------------------------------------------------
feathered_b64 = get_base64_image("assets/college_building_feathered.png")
if not feathered_b64:
    feathered_b64 = get_base64_image("assets/college_building.png")

img_tag = (
    f'<img src="{feathered_b64}" class="m3-building-img" alt="K. K. Wagh IEER Campus" />'
    if feathered_b64
    else ""
)

st.html(
    f"""
    <div class="m3-header-card">
        <div class="m3-header-left">
            <div class="m3-icon-badge">📈</div>
            <div>
                <h1 class="m3-header-title">Usage Patterns</h1>
                <div class="m3-header-subtitle">Understand daily and hourly energy usage across CSD Department spaces</div>
                <div class="m3-header-meta">
                    <span>🏛️ CSD Department &bull; E-Wing</span>
                    <span>📍 K. K. Wagh Institute of Engineering Education &amp; Research, Nashik</span>
                    <span>📅 Academic Year 2025</span>
                </div>
            </div>
        </div>
        <div class="m3-header-right">
            {img_tag}
        </div>
    </div>
    """
)

# -----------------------------------------------------------------------------
# 2. DEMO MODE Notice Banner
# -----------------------------------------------------------------------------
st.html(
    """
    <div class="demo-amber-banner">
        <span style="font-size: 1.05rem; line-height: 1;">⚠️</span>
        <span><strong>DEMO MODE</strong> &mdash; All electricity readings and results are simulated for academic demonstration. They are not actual college measurements.</span>
    </div>
    """
)

# -----------------------------------------------------------------------------
# 3. Interactive Filter Bar
# -----------------------------------------------------------------------------
df_raw = get_cached_dataset()

FLOOR_SPACES = {
    "All Floors": [
        "DSBD Lab",
        "DBMS Lab",
        "Classroom E501",
        "Classroom E502",
        "DSA Lab",
        "PBL Lab",
        "HOD Cabin",
        "Department Hall",
    ],
    "3rd Floor": ["DSBD Lab", "DBMS Lab"],
    "4th Floor": [
        "Classroom E501",
        "Classroom E502",
        "DSA Lab",
        "PBL Lab",
        "HOD Cabin",
        "Department Hall",
    ],
}

with st.container(border=True):
    f_col1, f_col2, f_col3 = st.columns([1.3, 0.95, 1.3], gap="medium")

    with f_col1:
        date_selection = st.date_input(
            "Date Range",
            value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
            min_value=datetime(2025, 1, 1),
            max_value=datetime(2025, 12, 31),
            help="Select evaluation period for usage pattern analysis",
        )
        if isinstance(date_selection, (tuple, list)):
            if len(date_selection) == 2:
                start_date, end_date = date_selection
            elif len(date_selection) == 1:
                start_date = end_date = date_selection[0]
            else:
                start_date, end_date = datetime(2025, 1, 1), datetime(2025, 12, 31)
        else:
            start_date = end_date = date_selection

    with f_col2:
        selected_floor = st.selectbox(
            "Floor",
            options=["All Floors", "3rd Floor", "4th Floor"],
            index=0,
            help="Filter monitored department floor levels",
        )

    valid_spaces = FLOOR_SPACES.get(selected_floor, FLOOR_SPACES["All Floors"])
    space_options = ["All Spaces (Combined)"] + valid_spaces

    with f_col3:
        selected_space = st.selectbox(
            "Monitored Space",
            options=space_options,
            index=0,
            help="Analyze diurnal and operating patterns for a specific space or across all monitored spaces",
        )

# Apply filters
is_all_spaces = (selected_space == "All Spaces (Combined)")
active_rooms = valid_spaces if is_all_spaces else [selected_space]
selected_floors_arg = None if selected_floor == "All Floors" else [selected_floor]

df_filtered = filter_dataset(
    df=df_raw,
    start_date=start_date,
    end_date=end_date,
    selected_rooms=active_rooms,
    selected_floors=selected_floors_arg,
)

if df_filtered.empty:
    st.warning("No electricity consumption records found for the selected filter combination.")
    st.stop()

# Ensure integer hour column is present
if "hour" not in df_filtered.columns:
    df_filtered["hour"] = df_filtered["timestamp"].dt.hour

# -----------------------------------------------------------------------------
# 4. Feature 1 Calculations: Hourly Profile & Daily Trend
# -----------------------------------------------------------------------------
# Hourly Average Profile (00:00 - 23:00)
hourly_agg = df_filtered.groupby("hour")["kwh"].mean()
# Ensure all 24 hours (0..23) are represented
hourly_full = pd.Series(0.0, index=range(24))
hourly_full.update(hourly_agg)

peak_hour = int(hourly_full.idxmax())
peak_val = float(hourly_full.max())
min_hour = int(hourly_full.idxmin())
min_val = float(hourly_full.min())

peak_hour_str = f"{peak_hour:02d}:00"
min_hour_str = f"{min_hour:02d}:00"
swing_ratio = (peak_val / max(min_val, 0.001)) if min_val > 0 else 0.0

# Daily Trend Aggregation
df_daily = df_filtered.groupby(df_filtered["timestamp"].dt.date)["kwh"].sum().reset_index()
df_daily.columns = ["date", "kwh"]
avg_daily_kwh = float(df_daily["kwh"].mean()) if not df_daily.empty else 0.0
total_days_evaluated = len(df_daily)

# Space title descriptor
space_label = selected_space if not is_all_spaces else f"All Spaces ({len(active_rooms)})"

# -----------------------------------------------------------------------------
# 5. Section 1: Daily and Hourly Usage Patterns
# -----------------------------------------------------------------------------
s1_title_col, s1_kpi1, s1_kpi2, s1_kpi3, s1_kpi4 = st.columns(
    [1.15, 0.72, 0.72, 0.75, 0.75], gap="small"
)

with s1_title_col:
    st.html(
        """
        <div style="display: flex; align-items: flex-start; gap: 10px; height: 100%; padding-top: 4px;">
            <div class="section-badge-circle">1</div>
            <div>
                <h2 class="sec-title-text">Daily and Hourly Usage Patterns</h2>
                <div class="sec-desc-text">Analyze typical 24-hour diurnal load curves and multi-day operational fluctuations.</div>
            </div>
        </div>
        """
    )

with s1_kpi1:
    st.html(
        f"""
        <div class="kpi-card kpi-card-green">
            <div class="kpi-icon-bubble bubble-green">⚡</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Peak Usage Hour</div>
                <div class="kpi-primary-val">{peak_hour_str}</div>
                <div class="kpi-secondary-text">{peak_val:.2f} kWh/hr</div>
            </div>
        </div>
        """
    )

with s1_kpi2:
    st.html(
        f"""
        <div class="kpi-card kpi-card-blue">
            <div class="kpi-icon-bubble bubble-blue">🌙</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Minimum Idle Hour</div>
                <div class="kpi-primary-val">{min_hour_str}</div>
                <div class="kpi-secondary-text">{min_val:.2f} kWh/hr</div>
            </div>
        </div>
        """
    )

with s1_kpi3:
    st.html(
        f"""
        <div class="kpi-card kpi-card-purple">
            <div class="kpi-icon-bubble bubble-purple">📅</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Daily Average Load</div>
                <div class="kpi-primary-val">{avg_daily_kwh:,.1f}</div>
                <div class="kpi-secondary-text">kWh/day ({total_days_evaluated}d)</div>
            </div>
        </div>
        """
    )

with s1_kpi4:
    st.html(
        f"""
        <div class="kpi-card kpi-card-peach">
            <div class="kpi-icon-bubble bubble-peach">📈</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Peak-to-Idle Ratio</div>
                <div class="kpi-primary-val">{swing_ratio:.1f}×</div>
                <div class="kpi-secondary-text">Diurnal Contrast</div>
            </div>
        </div>
        """
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Feature 1 Visualizations Row
s1_chart1, s1_chart2 = st.columns([1.55, 1.0], gap="medium")

with s1_chart1:
    with st.container(border=True):
        st.html(
            f"""
            <div class="chart-panel-header">
                <span>🕒</span>
                <span>Average Hourly Energy Profile (00:00 – 23:00) &bull; {space_label}</span>
            </div>
            """
        )

        hours_labels = [f"{h:02d}:00" for h in range(24)]
        fig_hourly = go.Figure()
        fig_hourly.add_trace(
            go.Scatter(
                x=hours_labels,
                y=hourly_full.values,
                mode="lines+markers",
                name="Average Hourly Demand",
                line=dict(color="#10B981", width=3, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.13)",
                marker=dict(
                    size=6,
                    color="#10B981",
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                hovertemplate="<b>Hour: %{x}</b><br>Average Demand: <b>%{y:.2f} kWh/hr</b><extra></extra>",
            )
        )

        # Highlight the peak hour with an annotation or point
        fig_hourly.add_trace(
            go.Scatter(
                x=[hours_labels[peak_hour]],
                y=[peak_val],
                mode="markers+text",
                name="Peak Hour",
                text=[f"Peak: {peak_val:.2f} kWh"],
                textposition="top center",
                textfont=dict(size=10.5, color="#065F46", family="sans-serif"),
                marker=dict(size=11, color="#EF4444", line=dict(width=2, color="#FFFFFF")),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        max_hourly_y = max(hourly_full.values) if len(hourly_full) > 0 else 5.0
        fig_hourly.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=45, r=20, t=20, b=40),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=10, color="#6B7280"),
                tickmode="linear",
                dtick=2,
            ),
            yaxis=dict(
                title=dict(
                    text="Average Rate (kWh/hr)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_hourly_y * 1.22],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_hourly, use_container_width=True, config={"displayModeBar": False})

with s1_chart2:
    with st.container(border=True):
        st.html(
            f"""
            <div class="chart-panel-header">
                <span>📅</span>
                <span>Daily Energy Consumption Trend &bull; {space_label}</span>
            </div>
            """
        )

        df_daily["date_str"] = pd.to_datetime(df_daily["date"]).dt.strftime("%d %b")
        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(
                x=df_daily["date_str"],
                y=df_daily["kwh"],
                mode="lines",
                name="Daily Energy",
                line=dict(color="#3B82F6", width=2, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.10)",
                hovertemplate="<b>Date: %{x}</b><br>Total Energy: <b>%{y:,.1f} kWh</b><extra></extra>",
            )
        )
        max_daily_y = float(df_daily["kwh"].max()) if not df_daily.empty else 100.0
        fig_daily.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=45, r=20, t=10, b=40),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=10, color="#6B7280"),
                nticks=8,
            ),
            yaxis=dict(
                title=dict(
                    text="Daily Energy (kWh)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_daily_y * 1.15],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_daily, use_container_width=True, config={"displayModeBar": False})

# Dynamic Observation Callout Box
st.html(
    f"""
    <div class="observation-box">
        <span style="font-size: 1.15rem; line-height: 1;">💡</span>
        <div>
            <strong>Hourly Pattern Observation:</strong>
            Average electricity demand for <strong>{space_label}</strong> peaks at <strong>{peak_hour_str}</strong> ({peak_val:.2f} kWh/hr), aligning with high daytime instructional and laboratory computing activity. Baseload standby electricity reaches its lowest baseline at <strong>{min_hour_str}</strong> ({min_val:.2f} kWh/hr), demonstrating a <strong>{swing_ratio:.1f}×</strong> diurnal operational swing between active daytime hours and overnight idle standby.
        </div>
    </div>
    """
)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Feature 2 Calculations: Academic Hours vs Off-Hours
# -----------------------------------------------------------------------------
# Configuration Controls for Configurable Academic Hours
s2_title_col, s2_cfg_col = st.columns([1.6, 1.0], gap="medium")

with s2_title_col:
    st.html(
        """
        <div style="display: flex; align-items: flex-start; gap: 10px; height: 100%; padding-top: 4px;">
            <div class="section-badge-circle">2</div>
            <div>
                <h2 class="sec-title-text">Academic Hours vs Off-Hours Comparison</h2>
                <div class="sec-desc-text">Contrast scheduled instructional periods against standby off-peak consumption.</div>
            </div>
        </div>
        """
    )

with s2_cfg_col:
    with st.container(border=True):
        c_sub1, c_sub2 = st.columns(2, gap="small")
        hour_labels_dict = {h: f"{h:02d}:00" for h in range(24)}
        with c_sub1:
            start_acad_hour = st.selectbox(
                "Academic Start",
                options=list(range(6, 13)),
                index=3,  # default 9 (09:00 AM)
                format_func=lambda h: f"{h:02d}:00 AM" if h < 12 else "12:00 PM",
                help="Start of assumed academic operating schedule",
            )
        with c_sub2:
            end_acad_hour = st.selectbox(
                "Academic End",
                options=list(range(14, 21)),
                index=3,  # default 17 (05:00 PM)
                format_func=lambda h: f"{h-12:02d}:00 PM" if h > 12 else f"{h:02d}:00 PM",
                help="End of assumed academic operating schedule",
            )

acad_hours_count = max(end_acad_hour - start_acad_hour, 1)
off_hours_count = 24 - acad_hours_count

# Classify records
df_filtered["period_type"] = "Off-Hours"
acad_mask = (df_filtered["hour"] >= start_acad_hour) & (df_filtered["hour"] < end_acad_hour)
df_filtered.loc[acad_mask, "period_type"] = "Academic Hours"

acad_sub = df_filtered[df_filtered["period_type"] == "Academic Hours"]
off_sub = df_filtered[df_filtered["period_type"] == "Off-Hours"]

acad_avg = float(acad_sub["kwh"].mean()) if not acad_sub.empty else 0.0
off_avg = float(off_sub["kwh"].mean()) if not off_sub.empty else 0.0
acad_total = float(acad_sub["kwh"].sum())
off_total = float(off_sub["kwh"].sum())
acad_cost = acad_total * tariff_rate
off_cost = off_total * tariff_rate
grand_total = acad_total + off_total
acad_pct = (acad_total / grand_total * 100) if grand_total > 0 else 0.0
off_pct = (off_total / grand_total * 100) if grand_total > 0 else 0.0

contrast_factor = (acad_avg / max(off_avg, 0.001)) if off_avg > 0 else 0.0
pct_diff = ((acad_avg - off_avg) / max(off_avg, 0.001) * 100) if off_avg > 0 else 0.0

# Descriptive configuration assumption banner
st.html(
    f"""
    <div class="config-assumption-card">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <span class="config-badge-academic">☀️ Academic Window: {start_acad_hour:02d}:00 &ndash; {end_acad_hour:02d}:00 ({acad_hours_count} hrs/day)</span>
            <span class="config-badge-off">🌙 Off-Hours Window: {end_acad_hour:02d}:00 &ndash; {start_acad_hour:02d}:00 ({off_hours_count} hrs/day)</span>
        </div>
        <div style="font-size: 0.74rem; color: #6B7280; font-style: italic;">
            * Configurable analytical assumption for laboratory investigation; costs estimated at ₹{tariff_rate:.2f}/kWh (simulated data).
        </div>
    </div>
    """
)

# 3 Feature 2 Highlight Cards
s2_kpi1, s2_kpi2, s2_kpi3 = st.columns(3, gap="medium")

with s2_kpi1:
    st.html(
        f"""
        <div class="kpi-card kpi-card-green">
            <div class="kpi-icon-bubble bubble-green">☀️</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Academic Hours Rate</div>
                <div class="kpi-primary-val">{acad_avg:.2f} <span style="font-size: 0.82rem; font-weight: 600;">kWh/hr</span></div>
                <div class="kpi-secondary-text">Total: {acad_total:,.0f} kWh &bull; Est. ₹{acad_cost:,.0f} ({acad_pct:.1f}%)</div>
            </div>
        </div>
        """
    )

with s2_kpi2:
    st.html(
        f"""
        <div class="kpi-card kpi-card-blue">
            <div class="kpi-icon-bubble bubble-blue">🌙</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Off-Hours Rate</div>
                <div class="kpi-primary-val">{off_avg:.2f} <span style="font-size: 0.82rem; font-weight: 600;">kWh/hr</span></div>
                <div class="kpi-secondary-text">Total: {off_total:,.0f} kWh &bull; Est. ₹{off_cost:,.0f} ({off_pct:.1f}%)</div>
            </div>
        </div>
        """
    )

with s2_kpi3:
    st.html(
        f"""
        <div class="kpi-card kpi-card-purple">
            <div class="kpi-icon-bubble bubble-purple">⚡</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">Operational Contrast</div>
                <div class="kpi-primary-val">{contrast_factor:.1f}× <span style="font-size: 0.82rem; font-weight: 600;">higher</span></div>
                <div class="kpi-secondary-text">+{pct_diff:,.0f}% daytime power surge</div>
            </div>
        </div>
        """
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Feature 2 Visualizations Row
s2_chart1, s2_chart2 = st.columns([1.0, 1.45], gap="medium")

with s2_chart1:
    with st.container(border=True):
        st.html(
            f"""
            <div class="chart-panel-header">
                <span>📊</span>
                <span>Average Hourly Rate: Academic vs Off-Hours</span>
            </div>
            """
        )

        compare_x = [
            f"Academic Hours<br>({start_acad_hour:02d}:00–{end_acad_hour:02d}:00)",
            f"Off-Hours<br>({end_acad_hour:02d}:00–{start_acad_hour:02d}:00)",
        ]
        compare_y = [acad_avg, off_avg]
        compare_colors = ["#10B981", "#60A5FA"]
        compare_text = [f"{acad_avg:.2f} kWh/hr", f"{off_avg:.2f} kWh/hr"]

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Bar(
                x=compare_x,
                y=compare_y,
                text=compare_text,
                textposition="outside",
                textfont=dict(size=12, color="#1F2937", family="sans-serif", weight="bold"),
                marker=dict(color=compare_colors, line=dict(width=0)),
                width=0.45,
                customdata=[acad_cost, off_cost],
                hovertemplate="<b>%{x}</b><br>Average Rate: <b>%{y:.2f} kWh/hr</b><br>Est. Period Cost (@ ₹" + f"{tariff_rate:.2f}/kWh): <b>₹%{{customdata:,.0f}}</b><extra></extra>",
            )
        )
        max_comp_y = max(compare_y) if compare_y else 5.0
        fig_compare.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=325,
            margin=dict(l=45, r=20, t=15, b=40),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#374151"),
            ),
            yaxis=dict(
                title=dict(
                    text="Average Rate (kWh/hr)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_comp_y * 1.25],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})

with s2_chart2:
    with st.container(border=True):
        st.html(
            """
            <div class="chart-panel-header">
                <span>🏢</span>
                <span>Space-Wise Contrast (Academic vs Off-Hours)</span>
            </div>
            """
        )

        # Space-wise comparison across active spaces
        comp_df = (
            df_filtered.groupby(["room_name", "period_type"])["kwh"]
            .mean()
            .unstack(fill_value=0.0)
        )

        # Sort spaces by academic hours descending
        if "Academic Hours" in comp_df.columns:
            comp_df = comp_df.sort_values(by="Academic Hours", ascending=False)

        space_names = list(comp_df.index)
        acad_vals = comp_df["Academic Hours"].values if "Academic Hours" in comp_df.columns else [0.0] * len(space_names)
        off_vals = comp_df["Off-Hours"].values if "Off-Hours" in comp_df.columns else [0.0] * len(space_names)

        fig_spaces = go.Figure()
        fig_spaces.add_trace(
            go.Bar(
                name="Academic Hours",
                x=space_names,
                y=acad_vals,
                marker=dict(color="#10B981"),
                hovertemplate="<b>%{x}</b><br>Academic Rate: <b>%{y:.2f} kWh/hr</b><extra></extra>",
            )
        )
        fig_spaces.add_trace(
            go.Bar(
                name="Off-Hours",
                x=space_names,
                y=off_vals,
                marker=dict(color="#60A5FA"),
                hovertemplate="<b>%{x}</b><br>Off-Hours Rate: <b>%{y:.2f} kWh/hr</b><extra></extra>",
            )
        )

        max_space_y = max(max(acad_vals), max(off_vals)) if len(acad_vals) > 0 else 5.0
        fig_spaces.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=325,
            margin=dict(l=45, r=20, t=15, b=45),
            barmode="group",
            bargap=0.25,
            bargroupgap=0.08,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10.5, color="#6B7280"),
            ),
            xaxis=dict(
                showgrid=False,
                linecolor="#E5E7EB",
                tickfont=dict(size=10, color="#374151"),
                tickangle=-15 if len(space_names) > 3 else 0,
            ),
            yaxis=dict(
                title=dict(
                    text="Average Rate (kWh/hr)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_space_y * 1.15],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_spaces, use_container_width=True, config={"displayModeBar": False})

# Short Interpretation Box
st.html(
    f"""
    <div class="academic-note-box">
        <strong>ℹ️ Academic Interpretation:</strong>
        Simulated electricity demand is approximately <strong>{contrast_factor:.1f}× higher</strong> during academic hours ({acad_avg:.2f} kWh/hr) compared to off-hours ({off_avg:.2f} kWh/hr). In computing laboratories (DSBD, DBMS, DSA, PBL), this sharp contrast reflects 25&ndash;30 concurrent student workstations and ventilation active during practicals, which shut down during off-hours to leave only standby idle baselines (~{off_avg:.2f} kWh/hr). In classrooms and faculty offices, the absolute loads are lower, but scheduled lighting and projection equipment drive a consistent daytime operational contrast.
    </div>
    """
)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Department Visual Context & Equipment Profile
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.html(
        """
        <div class="chart-panel-header">
            <span>🏛️</span>
            <span>Department Facility Context &amp; Equipment Assumptions</span>
        </div>
        """
    )

    lab_b64 = get_base64_image("assets/genuine_rooms/computer_lab.png")
    class_b64 = get_base64_image("assets/genuine_rooms/classroom.png")
    hod_b64 = get_base64_image("assets/genuine_rooms/hod_cabin.png")
    hall_b64 = get_base64_image("assets/genuine_rooms/dept_hall.png")

    ctx_col1, ctx_col2 = st.columns(2, gap="medium")

    with ctx_col1:
        st.html(
            f"""
            <div class="context-card">
                <div class="context-img-box">
                    <img src="{lab_b64}" alt="Computing Laboratory" />
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F3A2E; margin-bottom: 2px;">Computing Laboratories (4 Spaces)</div>
                    <div style="font-size: 0.77rem; color: #2D6A4F; font-weight: 600; margin-bottom: 4px;">DSBD Lab &bull; DBMS Lab &bull; DSA Lab &bull; PBL Lab</div>
                    <div style="font-size: 0.75rem; color: #6B7280; line-height: 1.4;">
                        High daytime energy intensity with 50 student &amp; faculty workstations, ceiling fans, LED lighting, and network switches. Operating hours drive a steep diurnal peak.
                    </div>
                </div>
            </div>
            """
        )

    with ctx_col2:
        st.html(
            f"""
            <div class="context-card">
                <div class="context-img-box">
                    <img src="{class_b64}" alt="Lecture Classroom" />
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0F3A2E; margin-bottom: 2px;">Lecture Classrooms &amp; Facilities (4 Spaces)</div>
                    <div style="font-size: 0.77rem; color: #2D6A4F; font-weight: 600; margin-bottom: 4px;">Classrooms E501, E502 &bull; HOD Cabin &bull; Dept. Hall</div>
                    <div style="font-size: 0.75rem; color: #6B7280; line-height: 1.4;">
                        Lecture and administrative spaces characterized by moderate daytime lighting, projection equipment, and fans, with near-zero idle standby power during overnight hours.
                    </div>
                </div>
            </div>
            """
        )
