"""
CSD Energy Intelligence - Module 1: Overview Dashboard
Individual Data Science & Big Data (DSBD) Laboratory Project
Department of Computer Science & Design (CSD)
K. K. Wagh Institute of Engineering Education & Research, Nashik
"""

from pathlib import Path
import streamlit as st

from src.ui_components import (
    inject_custom_css,
    render_pill_demo_banner,
    render_sidebar_department_info,
    render_sidebar_institutional_footer,
    render_sidebar_tariff_setting,
)

# Page configuration
st.set_page_config(
    page_title="CSD Energy Intelligence - Overview",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified academic styling
inject_custom_css()
render_sidebar_department_info()
selected_tariff = render_sidebar_tariff_setting()
render_sidebar_institutional_footer()

# Module 1 Specific Styling
st.html(
    """
    <style>
    /* Replace entrypoint 'app' text with 'Overview' in sidebar nav */
    [data-testid="stSidebarNavItems"] li:first-child a span {
        display: none !important;
    }
    [data-testid="stSidebarNavItems"] li:first-child a::after {
        content: "Overview" !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }

    /* Section titles */
    .section-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0F3A2E;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-subtitle {
        font-size: 0.84rem;
        color: #6B7280;
        margin-bottom: 16px;
    }

    /* Prototype status badge */
    .badge-prototype {
        background-color: #D8F3E5;
        color: #0F5132;
        border: 1px solid #B7E4C7;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }

    /* Objective Cards */
    .objective-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-top: 3px solid #1E754A;
    }
    .obj-num-pill {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: #E8F5E9;
        color: #1E754A;
        font-weight: 700;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .obj-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F3A2E;
        margin-bottom: 6px;
    }
    .obj-desc {
        font-size: 0.83rem;
        color: #4B5563;
        line-height: 1.45;
    }

    /* Information Grid Boxes */
    .info-box {
        background: #F8FAF9;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .info-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #52665D;
        margin-bottom: 3px;
    }
    .info-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F3A2E;
        line-height: 1.25;
        margin-bottom: 2px;
    }
    .info-subtext {
        font-size: 0.74rem;
        color: #6B7280;
    }

    /* Module Navigation Cards */
    .module-nav-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .module-badge-active {
        background: #D8F3E5;
        color: #0F5132;
        border: 1px solid #B7E4C7;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .module-badge-placeholder {
        background: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .module-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F3A2E;
        margin-bottom: 4px;
    }
    .module-desc {
        font-size: 0.81rem;
        color: #4B5563;
        line-height: 1.4;
        margin-bottom: 12px;
    }

    .image-caption-text {
        font-size: 0.78rem;
        color: #6B7280;
        text-align: center;
        margin-top: 6px;
        line-height: 1.35;
    }
    </style>
    """
)

# -----------------------------------------------------------------------------
# Section A: Institutional Header
# -----------------------------------------------------------------------------
st.html(
    f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: #E8F5E9; border: 1px solid #C8E6C9; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; flex-shrink: 0;">🌿</div>
                <div>
                    <h1 style="font-size: 1.52rem; font-weight: 800; color: #0F3A2E; margin: 0; line-height: 1.2; letter-spacing: -0.01em;">CSD Energy Intelligence</h1>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #2D6A4F; margin-top: 3px;">Smart Energy Consumption Analysis</div>
                    <div style="font-size: 0.82rem; color: #6B7280; margin-top: 2px;">CSD Department &nbsp;|&nbsp; K. K. Wagh Institute of Engineering Education & Research, Nashik</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="badge-prototype" style="background-color: #E0F2FE; color: #0369A1; border-color: #BAE6FD;">
                    <span>⚡</span> Tariff: ₹{selected_tariff:.2f} / kWh
                </div>
                <div class="badge-prototype">
                    <span>🧪</span> Prototype &nbsp;|&nbsp; Simulated Data
                </div>
            </div>
        </div>
    </div>
    """
)

# Mandatory Data Integrity Disclaimer Banner
render_pill_demo_banner()

# -----------------------------------------------------------------------------
# Section B: Project Introduction (with Energy Concept Image)
# -----------------------------------------------------------------------------
intro_col1, intro_col2 = st.columns([1.2, 1.0], gap="large")

with intro_col1:
    with st.container(border=True):
        st.html(
            """
            <div class="section-title">
                <span>🌿</span> Project Introduction
            </div>
            <div class="section-subtitle">
                Academic energy analytics prototype for the CSD department complex
            </div>
            <p style="font-size: 0.92rem; color: #374151; line-height: 1.65; margin-bottom: 16px;">
                CSD Energy Intelligence is an academic energy analytics prototype designed to explore electricity consumption patterns across classrooms, computer laboratories, faculty offices, and shared department spaces. It uses simulated hourly energy data and basic data science techniques to compare consumption, understand usage patterns, and highlight spaces with relatively high energy usage.
            </p>
            <div style="background: #F8FAF9; border-left: 3px solid #1E754A; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: #0F3A2E; margin-bottom: 4px;">Key Academic Highlights:</div>
                <div style="font-size: 0.80rem; color: #4B5563; line-height: 1.5;">
                    &bull; <strong>8 Monitored Spaces:</strong> 2 specialized labs on 3rd Floor; 2 classrooms, 2 labs, HOD cabin, and department hall on 4th Floor.<br>
                    &bull; <strong>Hourly Telemetry:</strong> Simulated load data capturing academic schedules, lab practicals, and standby idle baselines.<br>
                    &bull; <strong>Curriculum Alignment:</strong> Developed as an individual Data Science &amp; Big Data (DSBD) laboratory project.
                </div>
            </div>
            """
        )

with intro_col2:
    with st.container(border=True):
        st.html(
            """
            <div class="section-title">
                <span>💡</span> Energy Analytics Concept
            </div>
            <div class="section-subtitle">
                Visualizing data-driven campus energy awareness
            </div>
            """
        )
        concept_img = Path("assets/energy_concept.jpg")
        if concept_img.exists():
            st.image(str(concept_img), use_container_width=True)
            st.html(
                """
                <div class="image-caption-text">
                    <em>Concept Image:</em> Representing smart energy monitoring, efficiency analytics, and sustainable resource management in academic environments.
                </div>
                """
            )
        else:
            st.info("Energy Concept Image not found in assets.")

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section C: College Identity Section
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.html(
        """
        <div class="section-title">
            <span>🏛️</span> College Identity &amp; Campus Setting
        </div>
        <div class="section-subtitle">
            K. K. Wagh Institute of Engineering Education &amp; Research, Nashik
        </div>
        """
    )

    camp_col1, camp_col2 = st.columns([1.25, 1.0], gap="large")

    with camp_col1:
        college_img = Path("assets/college_building.png")
        if college_img.exists():
            st.image(str(college_img), use_container_width=True)
            st.html(
                """
                <div class="image-caption-text">
                    <strong>K. K. Wagh Institute of Engineering Education &amp; Research, Nashik</strong><br>
                    Campus building view. <em>Note: This photograph identifies the institutional campus; all department electricity readings displayed in this prototype are simulated.</em>
                </div>
                """
            )
        else:
            st.info("College building photograph not found in assets.")

    with camp_col2:
        st.html(
            """
            <div style="background: #F8FAF9; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F3A2E; margin-bottom: 6px;">
                        Department of Computer Science &amp; Design (CSD)
                    </div>
                    <div style="font-size: 0.82rem; color: #4B5563; line-height: 1.5; margin-bottom: 12px;">
                        The CSD department is situated across the 3rd and 4th floors of the academic complex, housing modern computing laboratories, lecture halls, faculty facilities, and shared study areas.
                    </div>
                    <div style="font-size: 0.80rem; color: #374151; line-height: 1.6; margin-bottom: 12px;">
                        <strong>Floor Distribution:</strong><br>
                        &bull; <strong style="color: #1E754A;">3rd Floor:</strong> DSBD Lab &bull; DBMS Lab<br>
                        &bull; <strong style="color: #1E754A;">4th Floor:</strong> Classroom E501 &bull; Classroom E502 &bull; DSA Lab &bull; PBL Lab &bull; HOD Cabin &bull; Department Hall
                    </div>
                </div>
                <div style="background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 8px 12px; font-size: 0.77rem; color: #64748B;">
                    <strong>Academic Context:</strong> The photograph illustrates the physical college campus. The prototype uses synthetic datasets to simulate electricity consumption across these spaces.
                </div>
            </div>
            """
        )

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section D: Project Objectives
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.html(
        """
        <div class="section-title">
            <span>🎯</span> Project Objectives
        </div>
        <div class="section-subtitle">
            Core analytical goals defined for the DSBD laboratory investigation
        </div>
        """
    )

    obj_col1, obj_col2, obj_col3 = st.columns(3, gap="medium")

    with obj_col1:
        st.html(
            """
            <div class="objective-box">
                <div>
                    <div class="obj-num-pill">1</div>
                    <div class="obj-title">Analyze Energy Consumption</div>
                    <div class="obj-desc">
                        Understand hourly and daily electricity usage patterns across lecture timings, practical sessions, evening hours, and weekends.
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.76rem; color: #1E754A; font-weight: 600;">
                    &check; Hourly &amp; daily trends
                </div>
            </div>
            """
        )

    with obj_col2:
        st.html(
            """
            <div class="objective-box">
                <div>
                    <div class="obj-num-pill">2</div>
                    <div class="obj-title">Compare Monitored Spaces</div>
                    <div class="obj-desc">
                        Compare energy consumption across labs, classrooms, the HOD cabin, and the department hall to evaluate load shares and equipment intensity.
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.76rem; color: #1E754A; font-weight: 600;">
                    &check; Multi-space benchmarking
                </div>
            </div>
            """
        )

    with obj_col3:
        st.html(
            """
            <div class="objective-box">
                <div>
                    <div class="obj-num-pill">3</div>
                    <div class="obj-title">Identify High Usage</div>
                    <div class="obj-desc">
                        Highlight spaces and time periods with comparatively high simulated consumption to support data-driven energy conservation awareness.
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.76rem; color: #1E754A; font-weight: 600;">
                    &check; High-usage space detection
                </div>
            </div>
            """
        )

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section E: Project Information
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.html(
        """
        <div class="section-title">
            <span>📋</span> Project Information &amp; Technical Scope
        </div>
        <div class="section-subtitle">
            Academic specifications and data parameters for the DSBD laboratory evaluation
        </div>
        """
    )

    inf_col1, inf_col2, inf_col3, inf_col4, inf_col5 = st.columns(5, gap="medium")

    with inf_col1:
        st.html(
            """
            <div class="info-box">
                <div class="info-label">Academic Domain</div>
                <div class="info-value">Data Science &amp; Big Data</div>
                <div class="info-subtext">DSBD Laboratory (TE CSD)</div>
            </div>
            """
        )

    with inf_col2:
        st.html(
            """
            <div class="info-box">
                <div class="info-label">Data Frequency</div>
                <div class="info-value">Hourly Records</div>
                <div class="info-subtext">8,760 hours per space (2025)</div>
            </div>
            """
        )

    with inf_col3:
        st.html(
            """
            <div class="info-box">
                <div class="info-label">Data Source</div>
                <div class="info-value">Simulated Dataset</div>
                <div class="info-subtext">Synthetic hourly academic model</div>
            </div>
            """
        )

    with inf_col4:
        st.html(
            """
            <div class="info-box">
                <div class="info-label">Primary Goal</div>
                <div class="info-value">Consumption Analysis</div>
                <div class="info-subtext">Pattern comparison &amp; insights</div>
            </div>
            """
        )

    with inf_col5:
        st.html(
            f"""
            <div class="info-box">
                <div class="info-label">Electricity Tariff</div>
                <div class="info-value">₹{selected_tariff:.2f} / kWh</div>
                <div class="info-subtext">Configurable in sidebar (default ₹11.00)</div>
            </div>
            """
        )

    st.html(
        """
        <div style="background: #F8FAF9; border-radius: 6px; padding: 10px 14px; margin-top: 14px; font-size: 0.79rem; color: #52665D; border: 1px solid #E5E7EB;">
            <strong>Data Integrity Statement:</strong> This project is developed purely for laboratory study. It uses simulated electricity readings based on documented equipment specifications (workstations, air conditioners, fans, fluorescent/LED lights) and timetable schedules. It does not connect to live sensors or physical college meters.
        </div>
        """
    )

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section F: Module Navigation
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.html(
        """
        <div class="section-title">
            <span>🚀</span> Application Modules
        </div>
        <div class="section-subtitle">
            The application consists of three main modules designed for data science exploration
        </div>
        """
    )

    nav_col1, nav_col2, nav_col3 = st.columns(3, gap="medium")

    with nav_col1:
        st.html(
            """
            <div class="module-nav-box" style="border-top: 3px solid #1E754A;">
                <div>
                    <span class="module-badge-active">&bull; Active Module</span>
                    <div class="module-name">Module 1: Overview</div>
                    <div class="module-desc">
                        Project introduction, campus identity, core objectives, and academic information.
                    </div>
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #1E754A; padding-top: 8px;">
                    &bull; Currently Viewing
                </div>
            </div>
            """
        )

    with nav_col2:
        with st.container():
            st.html(
                """
                <div class="module-nav-box" style="border-top: 3px solid #0369A1;">
                    <div>
                        <span class="module-badge-placeholder">Module 2 Placeholder</span>
                        <div class="module-name">Module 2: Energy Analysis</div>
                        <div class="module-desc">
                            Detailed energy consumption analysis, space comparisons, and high-usage space detection.
                        </div>
                    </div>
                </div>
                """
            )
            st.page_link(
                "pages/1_Energy_Analysis.py",
                label="Open Energy Analysis →",
                icon="📊",
            )

    with nav_col3:
        with st.container():
            st.html(
                """
                <div class="module-nav-box" style="border-top: 3px solid #D97706;">
                    <div>
                        <span class="module-badge-placeholder" style="background: #FEF3C7; color: #B45309; border-color: #FDE68A;">Module 3 Placeholder</span>
                        <div class="module-name">Module 3: Usage Patterns</div>
                        <div class="module-desc">
                            Temporal pattern analysis comparing academic operating hours against idle baselines.
                        </div>
                    </div>
                </div>
                """
            )
            st.page_link(
                "pages/2_Usage_Patterns.py",
                label="Open Usage Patterns →",
                icon="📈",
            )
