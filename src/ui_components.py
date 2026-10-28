"""
CSD Energy Intelligence - Shared UI Components & Design System
Maintains consistent dark green academic palette, clean cards, disclaimer banner, and Plotly styling.
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import load_energy_data
from src.preprocessing import create_ml_features

# Professional Academic Color Palette (CSD Energy Intelligence)
COLOR_PRIMARY_DARK = "#1B4332"
COLOR_PRIMARY = "#2D6A4F"
COLOR_ACCENT = "#40916C"
COLOR_ACCENT_LIGHT = "#74C69D"
COLOR_BG_CARD = "#F8F9FA"
COLOR_BORDER = "#E5E7EB"
COLOR_TEXT = "#1F2937"
COLOR_MUTED = "#6B7280"
COLOR_ALERT_HIGH = "#DC2626"
COLOR_ALERT_LOW = "#2563EB"
COLOR_WARNING = "#D97706"

# Exact Palette sampled from reference design (01_dashboard.png)
COLOR_SIDEBAR_BG = "#002D2A"
COLOR_NAV_ACTIVE = "#397F5D"
COLOR_NAV_TEXT = "#A7D8BE"
COLOR_CANVAS_BG = "#F4F8F5"
COLOR_MINT_ACCENT = "#1B7344"



def inject_custom_css():
    """Injects restrained, clean academic CSS styling into the Streamlit session."""
    st.markdown(
        f"""
        <style>
        /* Sidebar styling - Dark green academic aesthetic */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR_BG} !important;
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {{
            color: #E2E8F0 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.12) !important;
        }}

        /* Sidebar input styling for high contrast */
        section[data-testid="stSidebar"] input {{
            color: #111827 !important;
            background-color: #FFFFFF !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {{
            background-color: #FFFFFF !important;
            border-radius: 6px !important;
        }}
        section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
        section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] {{
            color: #111827 !important;
        }}

        /* Sidebar navigation link pills matching reference design */
        [data-testid="stSidebarNav"] {{
            padding-top: 0.5rem !important;
        }}
        [data-testid="stSidebarNavItems"] a {{
            color: {COLOR_NAV_TEXT} !important;
            font-weight: 500 !important;
            font-size: 0.90rem !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin-bottom: 2px !important;
            transition: all 0.15s ease-in-out !important;
        }}
        [data-testid="stSidebarNavItems"] a:hover {{
            background-color: rgba(57, 127, 93, 0.25) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebarNavItems"] a[aria-current="page"] {{
            background-color: {COLOR_NAV_ACTIVE} !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSidebarNavItems"] a[aria-current="page"] span {{
            color: #FFFFFF !important;
        }}

        /* Clean academic typography */
        body, .stMarkdown, p, div, span {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: {COLOR_TEXT};
        }}

        /* Soft mint-tinted canvas background */
        .stApp {{
            background-color: {COLOR_CANVAS_BG} !important;
        }}
        .main .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1320px;
        }}

        /* Disclaimer Banner (Standard Rectangular) */
        .demo-notice-banner {{
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 10px 16px;
            border-radius: 4px;
            font-size: 0.88rem;
            color: #92400E;
            font-weight: 500;
            margin-bottom: 20px;
        }}

        /* Sleek Mint Pill Disclaimer Banner (Matching 01_dashboard.png) */
        .demo-pill-banner {{
            background-color: #D8F3E5;
            border: 1px solid #B7E4C7;
            border-radius: 24px;
            padding: 7px 18px;
            color: #0F5132;
            font-size: 0.83rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        }}

        /* Department Header Card */
        .dept-header-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0 0 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            margin-bottom: 14px;
            overflow: hidden;
            height: 125px;
            min-height: 125px;
            position: relative;
        }}
        .dept-header-left {{
            display: flex;
            align-items: center;
            gap: 14px;
            z-index: 2;
            padding: 12px 0;
            flex-shrink: 0;
        }}
        .dept-logo-badge {{
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: #E8F5E9;
            border: 1px solid #C8E6C9;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.45rem;
            flex-shrink: 0;
        }}
        .dept-title-text {{
            font-size: 1.32rem;
            font-weight: 700;
            color: #0F3A2E;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -0.01em;
        }}
        .dept-subtitle-text {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #2D6A4F;
            margin-top: 3px;
        }}
        .dept-meta-text {{
            font-size: 0.78rem;
            color: #6B7280;
            margin-top: 2px;
        }}
        .dept-header-right {{
            display: flex;
            align-items: center;
            position: relative;
            height: 100%;
        }}
        .dept-date-pill {{
            position: absolute;
            top: 12px;
            right: 16px;
            z-index: 3;
            background: rgba(240, 250, 242, 0.94);
            border: 1px solid #C3E6CB;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #1E5E3A;
            white-space: nowrap;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            backdrop-filter: blur(4px);
        }}
        .dept-building-img {{
            height: 125px;
            width: auto;
            max-width: 440px;
            object-fit: cover;
            display: block;
        }}

        /* Legacy Header Card support */
        .dept-header-container {{
            background: #FFFFFF;
            border-bottom: 1px solid {COLOR_BORDER};
            padding-bottom: 14px;
            margin-bottom: 20px;
        }}
        .dept-title {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {COLOR_PRIMARY_DARK};
            margin: 0;
            padding: 0;
            letter-spacing: -0.02em;
        }}
        .dept-subtitle {{
            font-size: 0.92rem;
            color: {COLOR_MUTED};
            margin-top: 4px;
        }}

        /* KPI Metric Box */
        .kpi-card-box {{
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            height: 100%;
            min-height: 125px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-top-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }}
        .kpi-icon-pill {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            flex-shrink: 0;
        }}
        .kpi-header-label {{
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.2;
        }}
        .kpi-main-number {{
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 4px;
        }}
        .kpi-unit {{
            font-size: 1.05rem;
            font-weight: 600;
        }}
        .kpi-bottom-subtext {{
            font-size: 0.78rem;
            line-height: 1.35;
        }}
        .trend-badge-green {{
            display: inline-block;
            color: #1B7344;
            font-weight: 600;
            font-size: 0.78rem;
        }}

        /* Legacy Metric Cards */
        .metric-card {{
            background-color: #FFFFFF;
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            transition: border-color 0.15s ease;
        }}
        .metric-card:hover {{
            border-color: {COLOR_ACCENT_LIGHT};
        }}
        .metric-label {{
            font-size: 0.80rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLOR_MUTED};
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {COLOR_PRIMARY_DARK};
            line-height: 1.2;
        }}
        .metric-subtext {{
            font-size: 0.78rem;
            color: {COLOR_MUTED};
            margin-top: 4px;
        }}

        /* Granularity Radio Pills */
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            background: #F1F5F9;
            padding: 3px;
            border-radius: 20px;
            gap: 2px;
            display: inline-flex;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] label {{
            background: transparent;
            padding: 4px 14px !important;
            border-radius: 16px !important;
            margin: 0 !important;
            border: none !important;
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            color: #4B5563 !important;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {{
            background: #1E754A !important;
            color: #FFFFFF !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) span,
        div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {{
            color: #FFFFFF !important;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] label input {{
            display: none;
        }}

        /* Chart container card */
        .chart-card-wrapper {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}

        /* Recent Anomalies Table */
        .anomalies-card-wrapper {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-top: 14px;
        }}
        .anomalies-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.83rem;
        }}
        .anomalies-table th {{
            text-align: left;
            padding: 9px 12px;
            background: #F8FAF9;
            color: #52665D;
            font-weight: 600;
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            border-bottom: 1px solid #E5E7EB;
        }}
        .anomalies-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #F1F5F9;
            color: #1F2937;
            vertical-align: middle;
        }}
        .anomalies-table tr:hover td {{
            background-color: #F9FBFA;
        }}
        .badge-status-pending {{
            background: #FEF3C7;
            color: #B45309;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.74rem;
            font-weight: 600;
            display: inline-block;
        }}
        .badge-status-reviewed {{
            background: #D1FAE5;
            color: #065F46;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.74rem;
            font-weight: 600;
            display: inline-block;
        }}
        .table-action-pill {{
            background: #FFFFFF;
            border: 1px solid #D1D5DB;
            color: #374151;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 500;
            display: inline-block;
        }}

        /* Custom Badges */
        .badge-lab {{
            background-color: #DEF7EC;
            color: #03543F;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-classroom {{
            background-color: #E1EFFE;
            color: #1E429F;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-office {{
            background-color: #FDF2F8;
            color: #9D174D;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_demo_banner():
    """Renders the required prominent notice for simulated prototype data."""
    st.markdown(
        """
        <div class="demo-notice-banner">
            <strong>DEMO MODE</strong> — All electricity readings and results are simulated for academic demonstration. They are not actual college measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pill_demo_banner():
    """Renders sleek mint pill notice matching 01_dashboard.png."""
    st.html(
        '<div class="demo-pill-banner">'
        '<span style="font-size: 1.05rem; line-height: 1;">ⓘ</span>'
        '<span><strong>DEMO MODE</strong> &mdash; All electricity readings and results are simulated for academic demonstration. They are not actual college measurements.</span>'
        '</div>'
    )


def render_overview_header(
    start_d: Any,
    end_d: Any,
    image_path: str = "assets/e_wing_feathered.png",
):
    """Renders the primary institutional header card for Module 1 matching 01_dashboard.png."""
    img_tag = ""
    img_file = Path(image_path)
    if img_file.exists():
        try:
            with open(img_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            img_tag = f'<img src="data:image/png;base64,{b64}" class="dept-building-img" alt="CSD E-Wing Facilities" />'
        except Exception:
            img_tag = ""

    start_str = start_d.strftime("%d %b %Y") if hasattr(start_d, "strftime") else str(start_d)
    end_str = end_d.strftime("%d %b %Y") if hasattr(end_d, "strftime") else str(end_d)

    header_html = (
        '<div class="dept-header-card">'
        '<div class="dept-header-left">'
        '<div class="dept-logo-badge">🌿</div>'
        '<div>'
        '<div class="dept-title-text">CSD Energy Intelligence</div>'
        '<div class="dept-subtitle-text">CSD Department | E-Wing</div>'
        '<div class="dept-meta-text">3rd Floor &bull; 4th Floor &bull; 8 Monitored Spaces</div>'
        '</div>'
        '</div>'
        '<div class="dept-header-right">'
        f'<div class="dept-date-pill">📅 {start_str} &ndash; {end_str}</div>'
        f'{img_tag}'
        '</div>'
        '</div>'
    )
    st.html(header_html)


def render_page_header(page_title: str, subtitle: Optional[str] = None):
    """Renders standardized academic page header."""
    render_demo_banner()
    sub = subtitle if subtitle else "DSBD Laboratory Mini Project | Department of Computer Science & Design (E-Wing)"
    st.markdown(
        f"""
        <div class="dept-header-container">
            <h1 class="dept-title">{page_title}</h1>
            <div class="dept-subtitle">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_department_info():
    """Renders persistent departmental info and space guide in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 6px 0 14px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.35rem;">🌿</span>
                    <div>
                        <div style="color: #FFFFFF !important; font-size: 1.35rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.01em;">CSD</div>
                        <div style="font-size: 0.92rem; color: #A7D8BE !important; font-weight: 500;">Energy Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #A7D8BE !important; margin-bottom: 8px;">
                Monitored Spaces (8)
            </div>
            <div style="font-size: 0.80rem; color: #E2E8F0 !important; line-height: 1.45;">
                <strong style="color: #FFFFFF;">3rd Floor</strong>: DSBD Lab, DBMS Lab<br>
                <strong style="color: #FFFFFF;">4th Floor</strong>: Classroom E501, E502, DSA Lab, PBL Lab, HOD Cabin, Dept. Hall
            </div>
            <div style="font-size: 0.72rem; color: #CBD5E1 !important; line-height: 1.3; margin-top: 8px; padding: 6px 8px; background: rgba(255,255,255,0.06); border-radius: 4px;">
                <em>Note: Laboratory teacher seating areas (2 teachers + 1 TA) are metered as part of their respective labs.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")


def get_current_tariff() -> float:
    """Returns currently selected electricity tariff from session state, defaulting to ₹11.00/kWh."""
    if "tariff_rate" not in st.session_state:
        st.session_state["tariff_rate"] = 11.00
    return float(st.session_state["tariff_rate"])


def render_sidebar_tariff_setting() -> float:
    """
    Renders configurable electricity tariff input in sidebar.
    Allows values from ₹5.00 to ₹20.00 with ₹0.50 increments, defaulting to ₹11.00.
    Stores and synchronizes in st.session_state["tariff_rate"].
    """
    if "tariff_rate" not in st.session_state:
        st.session_state["tariff_rate"] = 11.00

    with st.sidebar:
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #A7D8BE !important; margin-bottom: 4px;">
                ⚙️ Tariff Setting
            </div>
            <div style="font-size: 0.74rem; color: #CBD5E1 !important; line-height: 1.35; margin-bottom: 6px;">
                Configurable electricity tariff (₹/kWh) for cost estimation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        val = st.number_input(
            "Tariff (₹/kWh)",
            min_value=5.00,
            max_value=20.00,
            value=float(st.session_state["tariff_rate"]),
            step=0.50,
            format="%.2f",
            key="tariff_rate_widget",
            help="Commercial academic electricity rate from ₹5 to ₹20 (default ₹11.00/kWh)",
        )
        st.session_state["tariff_rate"] = float(val)
        st.markdown(
            f"""
            <div style="font-size: 0.73rem; color: #82B29A !important; line-height: 1.3; margin-top: 2px; margin-bottom: 8px; padding: 4px 8px; background: rgba(255,255,255,0.06); border-radius: 4px;">
                Active Rate: <strong style="color: #FFFFFF;">₹{val:.2f} / kWh</strong> (Simulated)
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        return float(val)


def render_sidebar_institutional_footer():
    """Renders the college institutional footer at the bottom of the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size: 0.76rem; color: #82B29A !important; line-height: 1.4; padding: 12px 0 6px 0; border-top: 1px solid rgba(255,255,255,0.12); margin-top: 20px;">
                <strong style="color: #A7D8BE;">CSD Department</strong><br>
                E-Wing Facilities<br>
                K. K. Wagh Institute of Engineering Education & Research, Nashik
            </div>
            """,
            unsafe_allow_html=True,
        )



@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Loads and caches the simulated dataset."""
    csv_file = Path("data/csd_energy_simulated.csv")
    return load_energy_data(csv_file)


@st.cache_data(show_spinner=False)
def get_cached_ml_dataset() -> pd.DataFrame:
    """Pre-computes and caches ML lag and cyclical features for ML modules."""
    df = get_cached_dataset()
    return create_ml_features(df, include_lags=True)


def get_plotly_layout(title: str = "", height: int = 380) -> Dict[str, Any]:
    """Provides a consistent, publication-ready Plotly layout."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=COLOR_TEXT, family="sans-serif"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=45, r=25, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLOR_MUTED),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=11, color=COLOR_MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=11, color=COLOR_MUTED),
        ),
        font=dict(family="sans-serif", color=COLOR_TEXT),
    )
