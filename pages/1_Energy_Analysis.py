"""
CSD Energy Intelligence - Module 2: Energy Analysis
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
    page_title="Energy Analysis - CSD Energy Intelligence",
    page_icon="📊",
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
# Module 2 Custom CSS
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
    .m2-header-card {
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
    .m2-header-left {
        display: flex;
        align-items: center;
        gap: 16px;
        z-index: 2;
        padding: 12px 0;
        flex-shrink: 0;
    }
    .m2-icon-badge {
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
    .m2-header-title {
        font-size: 1.52rem;
        font-weight: 800;
        color: #0F3A2E;
        margin: 0;
        line-height: 1.18;
        letter-spacing: -0.01em;
    }
    .m2-header-subtitle {
        font-size: 0.88rem;
        font-weight: 600;
        color: #2D6A4F;
        margin-top: 3px;
    }
    .m2-header-meta {
        display: flex;
        align-items: center;
        gap: 18px;
        font-size: 0.77rem;
        color: #6B7280;
        margin-top: 5px;
        flex-wrap: wrap;
    }
    .m2-header-meta span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .m2-header-right {
        display: flex;
        align-items: center;
        position: relative;
        height: 100%;
    }
    .m2-building-img {
        height: 125px;
        width: auto;
        max-width: 440px;
        object-fit: cover;
        display: block;
    }

    /* DEMO MODE Notice Banner (Amber warning styled per reference image) */
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

    /* Section Header Components */
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
        padding: 8px 10px;
        display: flex;
        align-items: center;
        gap: 8px;
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
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    .bubble-green { background: #DCFCE7; color: #15803D; }
    .bubble-blue { background: #DBEAFE; color: #1D4ED8; }
    .bubble-purple { background: #F3E8FF; color: #7E22CE; }
    .bubble-peach { background: #FFEDD5; color: #C2410C; }
    .bubble-red { background: #FEE2E2; color: #DC2626; }

    .kpi-content-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
        flex: 1;
    }
    .kpi-label-title {
        font-size: 0.68rem;
        font-weight: 600;
        color: #6B7280;
        white-space: normal;
        line-height: 1.15;
        margin-bottom: 2px;
    }
    .kpi-primary-val {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0F3A2E;
        line-height: 1.15;
        white-space: nowrap;
    }
    .kpi-secondary-text {
        font-size: 0.70rem;
        color: #6B7280;
        white-space: nowrap;
    }

    /* Highlight Cards for Section 2 */
    .highlight-card-red {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .highlight-card-blue {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    /* Chart Container Card */
    .chart-panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        height: 100%;
    }
    .chart-panel-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.94rem;
        font-weight: 700;
        color: #0F3A2E;
    }

    /* Monitored Spaces 4x2 Gallery Grid */
    .room-gallery-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-top: 6px;
        margin-bottom: 10px;
    }
    .room-thumb-box {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 5px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .room-thumb-box:hover {
        border-color: #10B981;
        transform: translateY(-1px);
    }
    .room-thumb-crop {
        width: 100%;
        height: 58px;
        border-radius: 5px;
        overflow: hidden;
        background-color: #F3F4F6;
    }
    .room-thumb-crop img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .room-thumb-caption {
        font-size: 0.70rem;
        font-weight: 700;
        color: #1F2937;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Academic Interpretation Notice */
    .academic-note-box {
        background: #F8FAF9;
        border-left: 3px solid #1E754A;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.74rem;
        color: #4B5563;
        line-height: 1.42;
        margin-top: 6px;
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
    f'<img src="{feathered_b64}" class="m2-building-img" alt="K. K. Wagh IEER Campus" />'
    if feathered_b64
    else ""
)

st.html(
    f"""
    <div class="m2-header-card">
        <div class="m2-header-left">
            <div class="m2-icon-badge">📊</div>
            <div>
                <h1 class="m2-header-title">Energy Analysis</h1>
                <div class="m2-header-subtitle">Understand energy consumption patterns across CSD Department (E-Wing) spaces.</div>
                <div class="m2-header-meta">
                    <span>🏛️ CSD Department &bull; E-Wing</span>
                    <span>📍 K. K. Wagh Institute of Engineering Education &amp; Research, Nashik</span>
                    <span>📅 Academic Year 2025</span>
                </div>
            </div>
        </div>
        <div class="m2-header-right">
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

# Floor to space mappings
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
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.3, 0.95, 1.25, 0.9], gap="medium")

    with f_col1:
        date_selection = st.date_input(
            "Date Range",
            value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
            min_value=datetime(2025, 1, 1),
            max_value=datetime(2025, 12, 31),
            help="Select evaluation period for energy analytics",
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

    # Dynamic space choices based on floor
    valid_spaces = FLOOR_SPACES.get(selected_floor, FLOOR_SPACES["All Floors"])

    with f_col3:
        placeholder_label = "All Spaces (8)" if selected_floor == "All Floors" else f"All {selected_floor} Spaces"
        selected_spaces = st.multiselect(
            "Select Space(s)",
            options=valid_spaces,
            default=[],
            placeholder=placeholder_label,
            help="Choose specific department rooms or leave empty for all",
        )

    with f_col4:
        view_by = st.selectbox(
            "View By",
            options=["Monthly", "Daily", "Weekly"],
            index=0,
            help="Select aggregation granularity for trend and KPI metrics",
        )

# Apply filters
effective_spaces = selected_spaces if selected_spaces else valid_spaces
selected_floors_arg = None if selected_floor == "All Floors" else [selected_floor]

df_filtered = filter_dataset(
    df=df_raw,
    start_date=start_date,
    end_date=end_date,
    selected_rooms=effective_spaces,
    selected_floors=selected_floors_arg,
)

if df_filtered.empty:
    st.warning("No electricity consumption records found for the selected filter combination.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Core Calculations
# -----------------------------------------------------------------------------
total_kwh = float(df_filtered["kwh"].sum())
est_total_cost = total_kwh * tariff_rate

# Date formatting for labels
start_str = start_date.strftime("%b %Y") if hasattr(start_date, "strftime") else str(start_date)
end_str = end_date.strftime("%b %Y") if hasattr(end_date, "strftime") else str(end_date)
date_badge_str = f"{start_str} – {end_str}" if start_str != end_str else start_str

# Period Averages for Right Bar Chart
daily_grouped = df_filtered.groupby(df_filtered["timestamp"].dt.date)["kwh"].sum()
daily_avg = float(daily_grouped.mean()) if not daily_grouped.empty else 0.0

weekly_grouped = df_filtered.groupby(df_filtered["timestamp"].dt.to_period("W"))["kwh"].sum()
weekly_avg = float(weekly_grouped.mean()) if not weekly_grouped.empty else 0.0

monthly_grouped = df_filtered.groupby(df_filtered["timestamp"].dt.to_period("M"))["kwh"].sum()
monthly_avg = float(monthly_grouped.mean()) if not monthly_grouped.empty else 0.0

# Granular computations based on "View By"
if view_by == "Monthly":
    trend_df = (
        df_filtered.groupby(df_filtered["timestamp"].dt.to_period("M"))["kwh"]
        .sum()
        .reset_index()
    )
    trend_df["x_label"] = trend_df["timestamp"].apply(lambda p: p.strftime("%b %Y"))
    avg_kpi_label = "Average (Monthly)"
    avg_kpi_val = f"{monthly_avg:,.0f}"

    peak_row = trend_df.loc[trend_df["kwh"].idxmax()]
    peak_kpi_title = "Peak Month"
    peak_kpi_name = peak_row["x_label"]
    peak_kpi_sub = f"{peak_row['kwh']:,.0f} kWh"

    low_row = trend_df.loc[trend_df["kwh"].idxmin()]
    low_kpi_title = "Lowest Month"
    low_kpi_name = low_row["x_label"]
    low_kpi_sub = f"{low_row['kwh']:,.0f} kWh"
    trend_chart_title = "Monthly Energy Consumption Trend (All Spaces)" if not selected_spaces else f"Monthly Energy Consumption Trend ({', '.join(selected_spaces)})"

elif view_by == "Daily":
    trend_df = (
        df_filtered.groupby(df_filtered["timestamp"].dt.date)["kwh"]
        .sum()
        .reset_index()
    )
    trend_df["x_label"] = trend_df["timestamp"].apply(lambda d: pd.to_datetime(d).strftime("%d %b"))
    avg_kpi_label = "Average (Daily)"
    avg_kpi_val = f"{daily_avg:,.1f}"

    peak_row = trend_df.loc[trend_df["kwh"].idxmax()]
    peak_kpi_title = "Peak Day"
    peak_kpi_name = pd.to_datetime(peak_row["timestamp"]).strftime("%d %b %Y")
    peak_kpi_sub = f"{peak_row['kwh']:,.1f} kWh"

    low_row = trend_df.loc[trend_df["kwh"].idxmin()]
    low_kpi_title = "Lowest Day"
    low_kpi_name = pd.to_datetime(low_row["timestamp"]).strftime("%d %b %Y")
    low_kpi_sub = f"{low_row['kwh']:,.1f} kWh"
    trend_chart_title = "Daily Energy Consumption Trend (All Spaces)" if not selected_spaces else f"Daily Energy Consumption Trend ({', '.join(selected_spaces)})"

else:  # Weekly
    trend_df = (
        df_filtered.groupby(df_filtered["timestamp"].dt.to_period("W"))["kwh"]
        .sum()
        .reset_index()
    )
    trend_df["x_label"] = trend_df["timestamp"].apply(lambda p: f"W{p.week} ({p.start_time.strftime('%b')})")
    avg_kpi_label = "Average (Weekly)"
    avg_kpi_val = f"{weekly_avg:,.0f}"

    peak_row = trend_df.loc[trend_df["kwh"].idxmax()]
    peak_kpi_title = "Peak Week"
    peak_kpi_name = peak_row["x_label"]
    peak_kpi_sub = f"{peak_row['kwh']:,.0f} kWh"

    low_row = trend_df.loc[trend_df["kwh"].idxmin()]
    low_kpi_title = "Lowest Week"
    low_kpi_name = low_row["x_label"]
    low_kpi_sub = f"{low_row['kwh']:,.0f} kWh"
    trend_chart_title = "Weekly Energy Consumption Trend (All Spaces)" if not selected_spaces else f"Weekly Energy Consumption Trend ({', '.join(selected_spaces)})"

# Section 2 Room-wise Aggregation
room_totals = df_filtered.groupby("room_name")["kwh"].sum().sort_values(ascending=False)
highest_room_name = room_totals.index[0]
highest_room_val = room_totals.iloc[0]
highest_room_cost = highest_room_val * tariff_rate
lowest_room_name = room_totals.index[-1]
lowest_room_val = room_totals.iloc[-1]
lowest_room_cost = lowest_room_val * tariff_rate


# -----------------------------------------------------------------------------
# 5. Section 1: Energy Consumption Analysis
# -----------------------------------------------------------------------------
s1_col_title, s1_kpi1, s1_kpi2, s1_kpi3, s1_kpi4 = st.columns(
    [0.95, 0.78, 0.78, 0.80, 0.80], gap="small"
)

with s1_col_title:
    st.html(
        """
        <div style="display: flex; align-items: flex-start; gap: 10px; height: 100%; padding-top: 4px;">
            <div class="section-badge-circle">1</div>
            <div>
                <h2 class="sec-title-text">Energy Consumption Analysis</h2>
                <div class="sec-desc-text">Analyze daily, weekly and monthly electricity consumption trends.</div>
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
                <div class="kpi-label-title">Total Consumption</div>
                <div class="kpi-primary-val">{total_kwh:,.0f}</div>
                <div class="kpi-secondary-text">kWh &bull; Est. ₹{est_total_cost:,.0f}</div>
            </div>
        </div>
        """
    )

with s1_kpi2:
    st.html(
        f"""
        <div class="kpi-card kpi-card-blue">
            <div class="kpi-icon-bubble bubble-blue">📊</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">{avg_kpi_label}</div>
                <div class="kpi-primary-val">{avg_kpi_val}</div>
                <div class="kpi-secondary-text">kWh</div>
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
                <div class="kpi-label-title">{peak_kpi_title}</div>
                <div class="kpi-primary-val" style="font-size: 1.05rem;">{peak_kpi_name}</div>
                <div class="kpi-secondary-text">{peak_kpi_sub}</div>
            </div>
        </div>
        """
    )

with s1_kpi4:
    st.html(
        f"""
        <div class="kpi-card kpi-card-peach">
            <div class="kpi-icon-bubble bubble-peach">↗</div>
            <div class="kpi-content-col">
                <div class="kpi-label-title">{low_kpi_title}</div>
                <div class="kpi-primary-val" style="font-size: 1.05rem;">{low_kpi_name}</div>
                <div class="kpi-secondary-text">{low_kpi_sub}</div>
            </div>
        </div>
        """
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Section 1 Visualizations Row
s1_chart_col1, s1_chart_col2 = st.columns([1.75, 1.0], gap="medium")

with s1_chart_col1:
    with st.container(border=True):
        st.html(
            f"""
            <div class="chart-panel-header">
                <span>📈</span>
                <span>{trend_chart_title}</span>
            </div>
            """
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend_df["x_label"],
                y=trend_df["kwh"],
                mode="lines+markers",
                name="Consumption",
                line=dict(color="#10B981", width=3, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.14)",
                marker=dict(
                    size=7,
                    color="#10B981",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                hovertemplate="<b>%{x}</b><br>Consumption: <b>%{y:,.1f} kWh</b><extra></extra>",
            )
        )
        fig_trend.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=45, r=20, t=10, b=40),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
            ),
            yaxis=dict(
                title=dict(
                    text="Energy Consumption (kWh)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

with s1_chart_col2:
    with st.container(border=True):
        st.html(
            """
            <div class="chart-panel-header">
                <span>🕒</span>
                <span>Consumption by Period (All Spaces)</span>
            </div>
            """
        )

        period_names = ["Daily<br>(Avg)", "Weekly<br>(Avg)", "Monthly<br>(Avg)"]
        period_values = [daily_avg, weekly_avg, monthly_avg]
        period_colors = ["#6EE7B7", "#60A5FA", "#FB923C"]
        period_text = [f"{daily_avg:,.0f}", f"{weekly_avg:,.0f}", f"{monthly_avg:,.0f}"]

        fig_period = go.Figure()
        fig_period.add_trace(
            go.Bar(
                x=period_names,
                y=period_values,
                text=period_text,
                textposition="outside",
                textfont=dict(size=12, color="#1F2937", family="sans-serif"),
                marker=dict(color=period_colors, line=dict(width=0)),
                width=0.48,
                hovertemplate="<b>%{x}</b><br>Average: <b>%{y:,.1f} kWh</b><extra></extra>",
            )
        )
        max_period_y = max(period_values) if period_values else 100
        fig_period.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=45, r=20, t=15, b=40),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#4B5563"),
            ),
            yaxis=dict(
                title=dict(
                    text="Energy Consumption (kWh)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_period_y * 1.25],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_period, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Section 2: Room-wise Comparison & High-Usage Detection
# -----------------------------------------------------------------------------
s2_col_title, s2_high, s2_low = st.columns([1.5, 0.95, 0.95], gap="medium")

with s2_col_title:
    st.html(
        """
        <div style="display: flex; align-items: flex-start; gap: 10px; height: 100%; padding-top: 4px;">
            <div class="section-badge-circle">2</div>
            <div>
                <h2 class="sec-title-text">Room-wise Comparison &amp; High-Usage Detection</h2>
                <div class="sec-desc-text">Compare energy consumption across all monitored spaces and identify high-usage rooms.</div>
            </div>
        </div>
        """
    )

with s2_high:
    st.html(
        f"""
        <div class="highlight-card-red">
            <div class="kpi-icon-bubble bubble-red">↗</div>
            <div class="kpi-content-col">
                <div style="font-size: 0.72rem; font-weight: 700; color: #DC2626;">Highest Consumption</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F3A2E; line-height: 1.2;">{highest_room_name}</div>
                <div style="font-size: 0.75rem; color: #6B7280;">{highest_room_val:,.0f} kWh &bull; Est. ₹{highest_room_cost:,.0f}</div>
            </div>
        </div>
        """
    )

with s2_low:
    st.html(
        f"""
        <div class="highlight-card-blue">
            <div class="kpi-icon-bubble bubble-blue">↘</div>
            <div class="kpi-content-col">
                <div style="font-size: 0.72rem; font-weight: 700; color: #2563EB;">Lowest Consumption</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F3A2E; line-height: 1.2;">{lowest_room_name}</div>
                <div style="font-size: 0.75rem; color: #6B7280;">{lowest_room_val:,.0f} kWh &bull; Est. ₹{lowest_room_cost:,.0f}</div>
            </div>
        </div>
        """
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Section 2 Visualizations Row
s2_chart_col1, s2_chart_col2 = st.columns([1.45, 1.0], gap="medium")

with s2_chart_col1:
    with st.container(border=True):
        st.html(
            f"""
            <div class="chart-panel-header">
                <span>📊</span>
                <span>Room-wise Total Energy Consumption ({date_badge_str})</span>
            </div>
            """
        )

        ranked_rooms = list(room_totals.index)
        ranked_vals = list(room_totals.values)
        ranked_costs = [v * tariff_rate for v in ranked_vals]

        # Ranked color gradient palette matching reference design
        palette = [
            "#EF4444",  # Rank 1: Red
            "#F97316",  # Rank 2: Orange
            "#F59E0B",  # Rank 3: Amber
            "#FBBF24",  # Rank 4: Gold
            "#60A5FA",  # Rank 5: Light Blue
            "#38BDF8",  # Rank 6: Sky Blue
            "#818CF8",  # Rank 7: Periwinkle
            "#93C5FD",  # Rank 8: Soft Blue
        ]
        bar_colors = [palette[i % len(palette)] for i in range(len(ranked_rooms))]
        bar_labels = [f"{v:,.0f}" for v in ranked_vals]

        fig_rooms = go.Figure()
        fig_rooms.add_trace(
            go.Bar(
                x=ranked_rooms,
                y=ranked_vals,
                customdata=ranked_costs,
                text=bar_labels,
                textposition="outside",
                textfont=dict(size=11, color="#1F2937", family="sans-serif"),
                marker=dict(color=bar_colors, line=dict(width=0)),
                width=0.62,
                hovertemplate="<b>%{x}</b><br>Total Energy: <b>%{y:,.1f} kWh</b><br>Est. Cost (@ ₹" + f"{tariff_rate:.2f}/kWh): <b>₹%{{customdata:,.0f}}</b><extra></extra>",
            )
        )
        max_room_y = max(ranked_vals) if ranked_vals else 100
        fig_rooms.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=345,
            margin=dict(l=45, r=20, t=15, b=45),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                linecolor="#E5E7EB",
                tickfont=dict(size=10.5, color="#374151"),
                tickangle=-15,
            ),
            yaxis=dict(
                title=dict(
                    text="Energy Consumption (kWh)",
                    font=dict(size=11, color="#6B7280"),
                ),
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                tickfont=dict(size=11, color="#6B7280"),
                range=[0, max_room_y * 1.20],
            ),
            font=dict(family="sans-serif", color="#1F2937"),
        )
        st.plotly_chart(fig_rooms, use_container_width=True, config={"displayModeBar": False})

with s2_chart_col2:
    with st.container(border=True):
        st.html(
            """
            <div class="chart-panel-header">
                <span>🏛️</span>
                <span>Monitored Spaces (8)</span>
            </div>
            """
        )

        # 8 monitored spaces with their optimized thumbnails
        ordered_gallery_spaces = [
            ("DSBD Lab", "assets/gallery_rooms/dsbd_lab.jpg"),
            ("DBMS Lab", "assets/gallery_rooms/dbms_lab.jpg"),
            ("Classroom E501", "assets/gallery_rooms/classroom_e501.jpg"),
            ("Classroom E502", "assets/gallery_rooms/classroom_e502.jpg"),
            ("DSA Lab", "assets/gallery_rooms/dsa_lab.jpg"),
            ("PBL Lab", "assets/gallery_rooms/pbl_lab.jpg"),
            ("HOD Cabin", "assets/gallery_rooms/hod_cabin.jpg"),
            ("Department Hall", "assets/gallery_rooms/department_hall.jpg"),
        ]

        gallery_html_cards = []
        for space_name, img_path in ordered_gallery_spaces:
            b64_src = get_base64_image(img_path)
            gallery_html_cards.append(
                f"""
                <div class="room-thumb-box">
                    <div class="room-thumb-crop">
                        <img src="{b64_src}" alt="{space_name}" />
                    </div>
                    <div class="room-thumb-caption" title="{space_name}">{space_name}</div>
                </div>
                """
            )

        grid_html = f"""
        <div class="room-gallery-grid">
            {''.join(gallery_html_cards)}
        </div>
        <div class="academic-note-box">
            <strong>ℹ️ Academic Interpretation:</strong> Higher consumption in computing laboratories (DSBD, DBMS, DSA, PBL) stems from continuous loads of 25&ndash;30 desktop workstations and cooling during lab practicals. Classrooms and faculty spaces host lower equipment loads. High usage reflects functional capacity, not electrical wastage.
        </div>
        <div class="academic-note-box" style="margin-top: 6px; border-left-color: #0284C7; background: #F0F9FF;">
            <strong>⚡ Tariff &amp; Cost Estimation:</strong> Costs are estimated using the active rate of <strong>₹{tariff_rate:.2f}/kWh</strong> (Total: <strong>₹{est_total_cost:,.0f}</strong>). Values are simulated approximations for academic evaluation and do not represent actual college utility bills.
        </div>
        """
        st.html(grid_html)
