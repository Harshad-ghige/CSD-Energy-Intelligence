# CSD Energy Intelligence
> **DSBD Laboratory Mini Project | Department of Computer Science & Design (E-Wing)**  
> *Continuous Departmental Energy Analytics, Unsupervised Anomaly Detection, and Machine Learning Forecasting*

---

> [!IMPORTANT]
> **DEMO MODE & SIMULATION NOTICE**  
> All electricity consumption readings, temporal patterns, and analytical metrics presented in this application are generated using a reproducible synthetic equipment simulation model for academic demonstration. They are **not** actual physical college meter measurements. Estimated costs are illustrative what-if projections and do **not** represent an official institutional electricity bill.

---

## 1. Project Context & Objectives

The **CSD Energy Intelligence** dashboard is an engineering departmental energy analytics and machine learning system custom-tailored for the **Computer Science & Design (CSD) Department** located in the **E-Wing** of the engineering campus.

### Monitored Spatial Architecture (8 Spaces)
The department spans two physical levels in E-Wing across **8 monitored spaces**:

```
E-Wing CSD Department Infrastructure
├── Third Floor (Heavy Computing Facilities)
│   ├── 1. DSBD Lab (Data Science & Big Data Laboratory)
│   │      └── 50 PCs, 7 ceiling fans, 6 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating)
│   └── 2. DBMS Lab (Database Management Systems Laboratory)
│          └── 50 PCs, 6 ceiling fans, 5 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating)
│
└── Fourth Floor (Academic, Administrative & Event Facilities)
    ├── 3. Classroom E501 (65-seat lecture hall: 7 ceiling fans, 5 tube lights, 1 PC, 1 projector, 1 Wi-Fi router)
    ├── 4. Classroom E502 (65-seat lecture hall: 7 ceiling fans, 5 tube lights, 1 PC, 1 projector, 1 Wi-Fi router)
    ├── 5. DSA Lab (Data Structures & Algorithms Laboratory: 50 PCs, 6 fans, 6 tube lights, 1 projector, 1 router)
    ├── 6. PBL Lab (Project-Based Learning Lab: 50 PCs, 7 fans, 7 tube lights, 1 projector, 1 router)
    ├── 7. HOD Cabin (Department Head Office: 2 ceiling fans, 1 executive PC, 2 tube lights, 1 Wi-Fi router)
    └── 8. Department Hall (Multi-purpose Seminar & Event Hall: 8 fans, 1 PC, 1 projector, 7 tube lights, 1 PA speaker)
```

> [!NOTE]
> **Laboratory Seating Policy:** Each laboratory includes a dedicated teacher seating area accommodating two teachers and one technical assistant. These seating areas are metered directly as part of their respective laboratory space rather than on separate sub-meters.

---

## 2. Equipment Specifications & Energy Calculation Model

Rather than using static arbitrary base/peak values, consumption is calculated bottom-up from individual equipment specifications, ratings, and operational schedules:

$$\text{Energy (kWh)} = \frac{\text{Quantity} \times \text{Watts per Unit} \times \text{Operating Hours}}{1000}$$

### Verified Equipment Configuration

| Space | Monitored Equipment & Quantities | Active Power | Standby Power | Operating Schedule Assumption |
|---|---|---|---|---|
| **Classrooms (E501, E502)** | 7 Fans, 5 Tube Lights, 1 PC, 1 Projector, 1 Wi-Fi Router | 75 W / fan<br>28 W / light<br>120 W / PC<br>250 W / projector<br>18 W / router | 0 W<br>0 W<br>2 W<br>2 W<br>18 W (24/7) | 6.0 h/day (09:00–13:00, 14:00–16:00)<br>6.0 h/day<br>5.0 h/day<br>4.0 h/day<br>24.0 h/day (continuous) |
| **Labs (DSBD, DBMS, DSA, PBL)** | 50 Student & Faculty PCs, 6–7 Fans, 5–7 Tube Lights, 1 Projector, 1 Wi-Fi Router | 120–130 W / PC<br>75 W / fan<br>28 W / light<br>250 W / projector<br>18 W / router | 3 W / PC<br>0 W<br>0 W<br>2 W<br>18 W (24/7) | **6.0 h/day** (two 3h practical slots)<br>7.0 h/day<br>7.0 h/day<br>3.0–3.5 h/day<br>24.0 h/day (continuous) |
| **HOD Cabin** | 2 Fans, 1 Executive PC, 2 Tube Lights, 1 Wi-Fi Router | 75 W / fan<br>130 W / PC<br>28 W / light<br>18 W / router | 0 W<br>3 W<br>0 W<br>18 W (24/7) | 7.5 h/day (09:00–13:00, 14:00–17:30)<br>7.0 h/day<br>7.5 h/day<br>24.0 h/day (continuous) |
| **Department Hall** | 8 Fans, 1 Podium PC, 1 Event Projector, 7 Tube Lights, 1 PA Speaker System | 75 W / fan<br>120 W / PC<br>280 W / projector<br>28 W / light<br>150 W / speaker | 0 W<br>2 W<br>2 W<br>0 W<br>3 W | 3.0 h/day (seminars & club events)<br>3.0 h/day<br>3.0 h/day<br>3.5 h/day<br>3.0 h/day |

### Hourly Calculation & Standby Isolation
For each equipment item during hour $h$:
$$\text{Hourly kWh} = \frac{\text{Quantity} \times \left[ f_h \cdot \text{Watts}_{\text{active}} \cdot M_{\text{seasonal}} + (1 - f_h) \cdot \text{Watts}_{\text{standby}} \right]}{1000}$$
- Active and standby states are strictly mutually exclusive, preventing double-counting.
- $f_h \in [0.0, 1.0]$ represents the scheduled activity fraction.
- $M_{\text{seasonal}}$ represents the seasonal temperature factor (modulating fan intensity during summer vs. winter).
- Clamped to non-negative values ($kwh \ge 0.0$) with minor Gaussian electrical variation.
- Generating the full 2025 calendar year produces **70,080 hourly records** (8 spaces $\times$ 8,760 hours).

---

## 3. Technology Stack

- **Programming Language:** Python 3.11+
- **Web Application Framework:** Streamlit (v1.66+)
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Scientific Computing:** SciPy
- **Machine Learning Library:** Scikit-learn (`IsolationForest`, `RandomForestRegressor`, `KMeans`, `PCA`, `StandardScaler`)
- **Interactive Data Visualization:** Plotly Express & Plotly Graph Objects
- **Test Automation:** Python Native `unittest` + Streamlit `AppTest`

No external databases, Node.js runtimes, cloud subscriptions, or paid APIs are required. The entire application runs self-contained inside the local `.venv` environment.

---

## 4. Machine Learning Methodologies

### A. Unsupervised Anomaly Detection (`IsolationForest`)
- **Algorithm:** Scikit-learn `IsolationForest(n_estimators=100, contamination=0.015, random_state=42)`
- **Feature Set:** `['kwh', 'hour', 'day_of_week', 'is_weekend']`
- **Zero Label Leakage:** The simulated ground-truth anomaly flag `is_anomaly_ground_truth` is strictly excluded during model training. The detector operates in an unsupervised manner.
- **Baseline Modeling:** Calculates historical median and interquartile range (IQR) per space, hour, and day type to detect positive surges and abnormally low drops (e.g. breaker trips).
- **Interactive Audit Workflow:** Review status (`Pending Review`, `Reviewed`, `Explained`) and field notes are managed directly in the UI and persisted in `st.session_state`.
- **Domain Caution:** A statistical anomaly indicates an unexpected operational pattern (e.g., equipment left on overnight), not confirmed energy waste without an on-site facility audit.

### B. Consumption Prediction & 7-Day Forecasting (`RandomForestRegressor`)
- **Algorithm:** Scikit-learn `RandomForestRegressor(n_estimators=50, max_depth=14, min_samples_split=4, random_state=42)`
- **Engineered Features:**
  - Cyclical temporal features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
  - Operational indicators: `is_academic_hours`, `is_scheduled_occupancy`
  - Autoregressive lag features: `lag_24h` (same hour yesterday), `lag_168h` (same hour last week), and `rolling_mean_24h`
- **Chronological Split:** Strictly enforces chronological validation (80% train, subsequent 20% test) to prevent future data leakage.
- **Evaluation Metrics:** Un-fabricated Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ calculated from held-out test data.
- **Forward-Looking 7-Day Forecast:** Autoregressive 168-hour forward projection with uncertainty intervals based on academic timetables.
- **Explainability:** Gini feature importance analysis quantifying predictive factors.

### C. Behavioral Clustering (`StandardScaler`, `KMeans`, and `PCA`)
- **Feature Normalization:** Standardized using `StandardScaler` on behavioral consumption features:
  - Mean hourly load, peak load, median load, standard deviation
  - Nighttime consumption ratio (22:00–06:00 vs. total)
  - Weekend consumption ratio
  - Working hours average load (09:00–17:00 weekdays)
  - Peak-to-average ratio
- **Clustering:** `KMeans(n_clusters=3, random_state=42)` segments rooms into empirical operational archetypes (Heavy Intensive Labs, Moderate Working Spaces, Lecture Halls).
- **Dimensionality Reduction:** 2D projection via Principal Component Analysis (`PCA(n_components=2)`) for cluster visualization.
- **Objective Interpretation:** Groups operational profiles without assigning unsupported efficiency claims.

---

## 5. Dashboard Modules Overview

The application is structured into three focused academic modules:

| Module | File | Core Capabilities |
|---|---|---|
| **Module 1 — Overview Dashboard** | `app.py` | Institutional context, project objectives, technical scope, active tariff indicator, and visual campus overview. |
| **Module 2 — Energy Analysis** | `pages/1_Energy_Analysis.py` | Period aggregations (Daily, Weekly, Monthly), total consumption KPI & estimated costs, interactive filters (date range, floor, spaces), ranked room comparison chart, and 8-space visual gallery. |
| **Module 3 — Usage Patterns** | `pages/2_Usage_Patterns.py` | 24-hour diurnal demand profiles (00:00–23:00), daily consumption trends, dynamic hourly observations, configurable Academic Hours vs. Off-Hours operational contrast, and departmental equipment context. |

---

## 6. Configurable Electricity Tariff & Cost Estimation

### Default Tariff & UI Synchronization
- The default electricity tariff is set to **₹11.00 per kWh**.
- The tariff is configurable via a numeric input in the dark-green sidebar (`st.sidebar.number_input`, range ₹5.00 to ₹20.00 in ₹0.50 steps).
- The selected rate is synchronized across all modules via Streamlit session state (`st.session_state["tariff_rate"]`).

### Cost Estimation Formulas
1. **Estimated Total Electricity Cost:**
   $$\text{Estimated Cost (₹)} = \text{Total Energy (kWh)} \times \text{Tariff } (₹/\text{kWh})$$

2. **Room-Wise Estimated Cost:**
   $$\text{Room Cost (₹)} = \text{Room Energy (kWh)} \times \text{Tariff } (₹/\text{kWh})$$

3. **Academic vs. Off-Hours Cost:**
   $$\text{Academic Hours Cost (₹)} = \text{Academic Energy (kWh)} \times \text{Tariff } (₹/\text{kWh})$$
   $$\text{Off-Hours Cost (₹)} = \text{Off-Hours Energy (kWh)} \times \text{Tariff } (₹/\text{kWh})$$

> [!CAUTION]
> **Cost Estimation Notice:** All monetary figures are mathematical estimations derived from simulated prototype consumption. They illustrate potential financial impact for academic demonstration and do **not** represent an official institutional utility or electricity bill.

---

## 7. Installation & Execution Guide

### Step 1: Open Windows PowerShell & Activate Environment
```powershell
cd C:\Users\HP\Desktop\CSD-Energy-Intelligence
.\.venv\Scripts\Activate.ps1
```

### Step 2: (Optional) Deterministic Dataset Generation
To regenerate the reproducible 70,080-record dataset for calendar year 2025 across all 8 spaces:
```powershell
.\.venv\Scripts\python src/data_generator.py
```

### Step 3: Launch the Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit run app.py
```
The application will launch in your browser at:
`http://localhost:8501`

---

## 8. Running the Test Suite

The test suite validates data generation, mathematical calculations, ML models, and page rendering:

```powershell
# Run the complete test suite (16 tests)
.\.venv\Scripts\python -m unittest discover tests

# Or run individual test modules
.\.venv\Scripts\python -m unittest tests/test_calculations.py
.\.venv\Scripts\python -m unittest tests/test_pages_smoke.py
```

---

## 9. Academic Viva & Lab Defense Talking Points

- **Why an Equipment-Based Calculation Model?**  
  Calculating hourly energy bottom-up ($\text{Qty} \times \text{Watts} \times \text{Hours} / 1000$) provides physically explainable baseline data. Laboratory PCs drawing 120W active and 3W standby across a 6-hour daily schedule can be defended transparently in a viva.
- **Why Isolation Forest for Energy Anomaly Detection?**  
  Isolation Forest isolates anomalous data points instead of profiling normal clusters. Because energy surges and breaker trips are distinct and sparse, they require fewer random partition splits, making the algorithm fast ($O(n \log n)$) and robust on time-series distributions.
- **Why Chronological Train/Test Split Instead of K-Fold?**  
  Standard K-Fold randomly shuffles timestamps, causing temporal data leakage (future data predicts past data). Chronological splitting respects the arrow of time, replicating real-world deployment.
- **Why Withhold Ground Truth Labels from Isolation Forest?**  
  In live facilities, electrical faults and human oversights are unlabelled. Isolation Forest operates strictly unsupervised; ground-truth anomaly flags are reserved exclusively for academic evaluation.
- **Seating Area Metering Rationale:**  
  Teacher seating areas inside laboratories (2 teachers + 1 TA) share electrical sub-distribution circuits with their respective labs. Modeling them within each lab reflects standard departmental wiring.

---

## 10. Limitations & Future Scope

1. **Simulated Prototype Data:** Readings are generated using equipment wattage assumptions and timetable schedules rather than live smart meter hardware.
2. **Sub-Metering Granularity:** Readings represent space-level aggregation. Future work can implement Non-Intrusive Load Monitoring (NILM) to disaggregate individual workstation or appliance consumption.
3. **Hardware Integration:** The modular data layer can be connected to an MQTT or Modbus IoT pipeline to ingest live college meter telemetry without modifying dashboard analytics.
