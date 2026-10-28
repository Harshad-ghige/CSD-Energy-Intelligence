"""
CSD Energy Intelligence - Synthetic Data Generator
Generates realistic simulated electricity consumption data for the CSD Department (E-Wing).

Monitored Spaces (8 Total):
  Third Floor:
    1. DSBD Lab (Data Science & Big Data Lab)
    2. DBMS Lab (Database Management Systems Lab)
  Fourth Floor:
    3. Classroom E501
    4. Classroom E502
    5. DSA Lab (Data Structures & Algorithms Lab)
    6. PBL Lab (Project-Based Learning Lab)
    7. HOD Cabin (Head of Department Office)
    8. Department Hall (Multi-purpose Seminar & Event Hall)

Note: Teacher seating areas (2 teachers + 1 tech assistant) are included within each laboratory.
All readings, equipment ratings, and schedules are strictly simulated for academic demonstration.
They are not verified college measurements.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Global Seed for full reproducibility
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Monitored Spaces & Equipment Configurations
# ---------------------------------------------------------------------------
# Equipment assumptions:
# - Classroom: 7 fans, 5 tube lights, 1 PC, 1 projector, 1 Wi-Fi router.
# - Lab base: 50 PCs, 6-7 fans, 5-7 tube lights, 1 projector, 1 Wi-Fi router.
# - HOD Cabin: 2 fans, 1 PC, 2 tube lights, 1 Wi-Fi router.
# - Department Hall: 8 fans, 1 PC, 1 projector, 7 tube lights, 1 speaker.
# Lab PC usage target: 6 hours/day during academic lab slots.
# Assumed representative wattages (illustrative academic prototype):
# - PC active: 120W - 130W, standby: 3W
# - Ceiling Fan: 75W, standby: 0W
# - LED Tube Light: 28W, standby: 0W
# - Projector: 250W - 280W, standby: 2W
# - Wi-Fi Router: 18W (24/7 continuous operation)
# - PA Speaker System: 150W, standby: 3W
# ---------------------------------------------------------------------------

MONITORED_SPACES = [
    # --- THIRD FLOOR ---
    {
        "room_id": "DSBD_LAB",
        "room_name": "DSBD Lab",
        "floor": "3rd Floor",
        "room_type": "Laboratory",
        "description": "50 PCs, 7 ceiling fans, 6 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating for 2 teachers + 1 TA)",
        "is_lab": True,
        "ac_equipped": False,
        "equipment": [
            {"name": "Student & Faculty PCs", "category": "lab_pcs", "quantity": 50, "watts_per_unit": 130, "standby_watts_per_unit": 3.0, "daily_operating_hours": 6.0},
            {"name": "Ceiling Fans", "category": "fans", "quantity": 7, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 6, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.5},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "DBMS_LAB",
        "room_name": "DBMS Lab",
        "floor": "3rd Floor",
        "room_type": "Laboratory",
        "description": "50 PCs, 6 ceiling fans, 5 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating for 2 teachers + 1 TA)",
        "is_lab": True,
        "ac_equipped": False,
        "equipment": [
            {"name": "Student & Faculty PCs", "category": "lab_pcs", "quantity": 50, "watts_per_unit": 120, "standby_watts_per_unit": 3.0, "daily_operating_hours": 6.0},
            {"name": "Ceiling Fans", "category": "fans", "quantity": 6, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 5, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.0},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    # --- FOURTH FLOOR ---
    {
        "room_id": "CLASS_E501",
        "room_name": "Classroom E501",
        "floor": "4th Floor",
        "room_type": "Classroom",
        "description": "65-seat lecture hall with 7 ceiling fans, 5 LED tube lights, 1 PC, 1 projector, 1 Wi-Fi router",
        "is_lab": False,
        "ac_equipped": False,
        "equipment": [
            {"name": "Ceiling Fans", "category": "fans", "quantity": 7, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 6.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 5, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 6.0},
            {"name": "Faculty PC", "category": "classroom_pc", "quantity": 1, "watts_per_unit": 120, "standby_watts_per_unit": 2.0, "daily_operating_hours": 5.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 4.0},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "CLASS_E502",
        "room_name": "Classroom E502",
        "floor": "4th Floor",
        "room_type": "Classroom",
        "description": "65-seat lecture hall with 7 ceiling fans, 5 LED tube lights, 1 PC, 1 projector, 1 Wi-Fi router",
        "is_lab": False,
        "ac_equipped": False,
        "equipment": [
            {"name": "Ceiling Fans", "category": "fans", "quantity": 7, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 6.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 5, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 6.0},
            {"name": "Faculty PC", "category": "classroom_pc", "quantity": 1, "watts_per_unit": 120, "standby_watts_per_unit": 2.0, "daily_operating_hours": 5.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 4.0},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "DSA_LAB",
        "room_name": "DSA Lab",
        "floor": "4th Floor",
        "room_type": "Laboratory",
        "description": "50 PCs, 6 ceiling fans, 6 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating for 2 teachers + 1 TA)",
        "is_lab": True,
        "ac_equipped": False,
        "equipment": [
            {"name": "Student & Faculty PCs", "category": "lab_pcs", "quantity": 50, "watts_per_unit": 120, "standby_watts_per_unit": 3.0, "daily_operating_hours": 6.0},
            {"name": "Ceiling Fans", "category": "fans", "quantity": 6, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 6, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.0},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "PBL_LAB",
        "room_name": "PBL Lab",
        "floor": "4th Floor",
        "room_type": "Laboratory",
        "description": "50 PCs, 7 ceiling fans, 7 LED tube lights, 1 projector, 1 Wi-Fi router (includes teacher seating for 2 teachers + 1 TA)",
        "is_lab": True,
        "ac_equipped": False,
        "equipment": [
            {"name": "Student & Faculty PCs", "category": "lab_pcs", "quantity": 50, "watts_per_unit": 120, "standby_watts_per_unit": 3.0, "daily_operating_hours": 6.0},
            {"name": "Ceiling Fans", "category": "fans", "quantity": 7, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 7, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.0},
            {"name": "Digital Projector", "category": "projector", "quantity": 1, "watts_per_unit": 250, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.0},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "HOD_CABIN",
        "room_name": "HOD Cabin",
        "floor": "4th Floor",
        "room_type": "HOD Cabin / Office",
        "description": "Department Head office with 2 ceiling fans, 1 executive PC, 2 LED tube lights, 1 Wi-Fi router",
        "is_lab": False,
        "ac_equipped": False,
        "equipment": [
            {"name": "Ceiling Fans", "category": "fans", "quantity": 2, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.5},
            {"name": "Executive PC", "category": "office_pc", "quantity": 1, "watts_per_unit": 130, "standby_watts_per_unit": 3.0, "daily_operating_hours": 7.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 2, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 7.5},
            {"name": "Wi-Fi Router", "category": "continuous", "quantity": 1, "watts_per_unit": 18, "standby_watts_per_unit": 18.0, "daily_operating_hours": 24.0},
        ],
    },
    {
        "room_id": "DEPT_HALL",
        "room_name": "Department Hall",
        "floor": "4th Floor",
        "room_type": "Department Hall",
        "description": "Multi-purpose seminar & event hall with 8 ceiling fans, 1 PC, 1 projector, 7 LED tube lights, 1 PA speaker system",
        "is_lab": False,
        "ac_equipped": False,
        "equipment": [
            {"name": "Ceiling Fans", "category": "fans", "quantity": 8, "watts_per_unit": 75, "standby_watts_per_unit": 0.0, "daily_operating_hours": 3.0},
            {"name": "Podium PC", "category": "classroom_pc", "quantity": 1, "watts_per_unit": 120, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.0},
            {"name": "Event Projector", "category": "projector", "quantity": 1, "watts_per_unit": 280, "standby_watts_per_unit": 2.0, "daily_operating_hours": 3.0},
            {"name": "LED Tube Lights", "category": "lighting", "quantity": 7, "watts_per_unit": 28, "standby_watts_per_unit": 0.0, "daily_operating_hours": 3.5},
            {"name": "PA Speaker System", "category": "speaker", "quantity": 1, "watts_per_unit": 150, "standby_watts_per_unit": 3.0, "daily_operating_hours": 3.0},
        ],
    },
]

# Calculate equipment-derived baseline standby and peak operating ratings for backward compatibility
for space in MONITORED_SPACES:
    space["base_standby_kwh"] = round(
        sum(e["quantity"] * e.get("standby_watts_per_unit", 0.0) for e in space["equipment"]) / 1000.0, 3
    )
    space["peak_operating_kwh"] = round(
        sum(e["quantity"] * e["watts_per_unit"] for e in space["equipment"]) / 1000.0, 3
    )


# ---------------------------------------------------------------------------
# Vacation & Seasonal Helpers
# ---------------------------------------------------------------------------

def is_college_vacation(date: pd.Timestamp) -> bool:
    """Check if the date falls in summer break (May 20 to June 25) or winter break (Dec 22 to Dec 31)."""
    month = date.month
    day = date.day
    # Summer vacation
    if (month == 5 and day >= 20) or (month == 6 and day <= 25):
        return True
    # Winter vacation
    if month == 12 and day >= 22:
        return True
    return False


def get_seasonal_temperature_factor(month: int) -> float:
    """
    Simulate seasonal temperature multiplier for an Indian climate:
    Summer (March to June): Higher multiplier for fan operating speed and ambient cooling.
    Monsoon (July to September): Moderate.
    Winter (December to February): Low temperature, reduced fan usage.
    """
    season_weights = {
        1: 0.82,   # Jan - Cool winter
        2: 0.88,   # Feb - Mild
        3: 1.10,   # Mar - Getting warm
        4: 1.35,   # Apr - Peak summer
        5: 1.40,   # May - Peak summer
        6: 1.25,   # Jun - Monsoon onset
        7: 1.10,   # Jul - Monsoon humid
        8: 1.05,   # Aug - Monsoon
        9: 1.08,   # Sep - Post-monsoon warm
        10: 1.00,  # Oct - Autumn
        11: 0.88,  # Nov - Mild
        12: 0.80,  # Dec - Winter
    }
    return season_weights.get(month, 1.0)


# ---------------------------------------------------------------------------
# Equipment-Based Energy Calculations
# ---------------------------------------------------------------------------

def calculate_equipment_energy_kwh(
    quantity: int,
    watts_per_unit: float,
    operating_hours: float,
) -> float:
    """
    Core project equipment energy formula:
    energy_kwh = quantity * watts_per_unit * operating_hours / 1000.0
    """
    return (quantity * watts_per_unit * operating_hours) / 1000.0


def compute_hourly_equipment_kwh(
    equipment_item: Dict[str, Any],
    active_fraction: float,
    temp_factor: float = 1.0,
) -> float:
    """
    Calculates hourly energy (kWh) for a single equipment entry.
    - Active energy for the active portion of the hour:
        active_kwh = quantity * (watts_per_unit * multiplier) * active_fraction / 1000.0
    - Standby energy for the remaining non-active portion:
        standby_kwh = quantity * standby_watts_per_unit * (1.0 - active_fraction) / 1000.0

    Guarantees active and standby states are mutually exclusive; no double-counting.
    """
    qty = equipment_item["quantity"]
    active_watts = equipment_item["watts_per_unit"]
    standby_watts = equipment_item.get("standby_watts_per_unit", 0.0)

    # Temperature factor modulates fan operating intensity
    multiplier = temp_factor if equipment_item.get("category") == "fans" else 1.0

    active_kwh = (qty * active_watts * multiplier * active_fraction) / 1000.0
    standby_kwh = (qty * standby_watts * (1.0 - active_fraction)) / 1000.0
    return active_kwh + standby_kwh


def get_equipment_active_fractions(
    room_id: str,
    room_type: str,
    hour: int,
    day_of_week: int,
    in_vacation: bool,
) -> Tuple[Dict[str, float], int]:
    """
    Returns (active_fractions_dict, is_scheduled_occupancy) for each equipment category in the given hour.
    Explicit editable schedules for weekdays, Saturdays, Sundays, and college vacations.
    """
    is_sunday = (day_of_week == 6)
    is_saturday = (day_of_week == 5)
    is_weekday = (day_of_week < 5)

    # Default: everything off/standby, continuous devices run 1.0
    fractions: Dict[str, float] = {
        "lab_pcs": 0.0,
        "fans": 0.0,
        "lighting": 0.0,
        "projector": 0.0,
        "classroom_pc": 0.0,
        "office_pc": 0.0,
        "speaker": 0.0,
        "continuous": 1.0,
    }
    scheduled_occupancy = 0

    if room_type == "Laboratory":
        if is_weekday and not in_vacation:
            # 6 Hours/day of lab PC practical usage: 09:00-12:00 (3h) and 14:00-17:00 (3h)
            if (9 <= hour < 12) or (14 <= hour < 17):
                scheduled_occupancy = 1
                fractions["lab_pcs"] = 1.0
                fractions["fans"] = 1.0
                fractions["lighting"] = 1.0
                fractions["projector"] = 0.6 if hour in (9, 14) else 0.2  # initial lab demos
            elif hour == 12:
                # Faculty discussion / lab wrap-up
                fractions["fans"] = 0.3
                fractions["lighting"] = 0.4
            elif hour == 13:
                # Lunch break (lab closed, TA in seating area)
                fractions["fans"] = 0.15
                fractions["lighting"] = 0.15
            elif hour in (8, 17):
                # Lab setup / winding down
                fractions["fans"] = 0.25
                fractions["lighting"] = 0.3
            elif hour == 18:
                fractions["fans"] = 0.15
                fractions["lighting"] = 0.2
        elif is_saturday and not in_vacation:
            # Saturday mentoring / remedial: 09:00 - 12:00 (3h, reduced PC utilization)
            if 9 <= hour < 12:
                scheduled_occupancy = 1
                fractions["lab_pcs"] = 0.4
                fractions["fans"] = 0.6
                fractions["lighting"] = 0.6
            elif hour in (8, 12):
                fractions["fans"] = 0.2
                fractions["lighting"] = 0.2
        elif in_vacation and is_weekday:
            # Vacation maintenance: 10:00 - 14:00
            if 10 <= hour < 14:
                fractions["lab_pcs"] = 0.05
                fractions["fans"] = 0.3
                fractions["lighting"] = 0.3

    elif room_type == "Classroom":
        if is_weekday and not in_vacation:
            # Lecture slots: 09:00-13:00 (4h) and 14:00-16:00 (2h) = 6 lecture hours
            if (9 <= hour < 13) or (14 <= hour < 16):
                scheduled_occupancy = 1
                fractions["fans"] = 1.0
                fractions["lighting"] = 1.0
                fractions["classroom_pc"] = 0.85
                fractions["projector"] = 0.70
            elif hour == 13:
                # Lunch break
                fractions["fans"] = 0.1
                fractions["lighting"] = 0.1
            elif hour in (8, 16):
                fractions["fans"] = 0.2
                fractions["lighting"] = 0.2
        elif is_saturday and not in_vacation:
            # Saturday tutorial: 09:00 - 12:00
            if 9 <= hour < 12:
                scheduled_occupancy = 1
                fractions["fans"] = 0.6
                fractions["lighting"] = 0.6
                fractions["classroom_pc"] = 0.5
                fractions["projector"] = 0.5

    elif room_type == "HOD Cabin / Office":
        if is_weekday and not in_vacation:
            # Office hours: 09:00-13:00 (4h) and 14:00-17:30 (3.5h) = ~7.5h
            if (9 <= hour < 13) or (14 <= hour < 18):
                scheduled_occupancy = 1
                fractions["fans"] = 0.95
                fractions["lighting"] = 0.95
                fractions["office_pc"] = 0.90
            elif hour == 13:
                # Lunch
                fractions["fans"] = 0.3
                fractions["lighting"] = 0.3
                fractions["office_pc"] = 0.2
            elif hour in (8, 18):
                fractions["fans"] = 0.2
                fractions["lighting"] = 0.2
        elif is_saturday and not in_vacation:
            if 9 <= hour < 13:
                scheduled_occupancy = 1
                fractions["fans"] = 0.7
                fractions["lighting"] = 0.7
                fractions["office_pc"] = 0.6
        elif in_vacation and is_weekday:
            if 10 <= hour < 15:
                scheduled_occupancy = 1
                fractions["fans"] = 0.5
                fractions["lighting"] = 0.5
                fractions["office_pc"] = 0.5

    elif room_type == "Department Hall":
        if is_weekday and not in_vacation:
            # Seminars, guest lectures, club sessions on selected weekdays (Tue=1, Wed=2, Fri=4) 14:00-17:00
            if (day_of_week in (1, 2, 4)) and (14 <= hour < 17):
                scheduled_occupancy = 1
                fractions["fans"] = 1.0
                fractions["lighting"] = 1.0
                fractions["classroom_pc"] = 1.0
                fractions["projector"] = 1.0
                fractions["speaker"] = 0.85
            # Monday morning briefing / meeting (10:00-12:00)
            elif (day_of_week == 0) and (10 <= hour < 12):
                scheduled_occupancy = 1
                fractions["fans"] = 0.8
                fractions["lighting"] = 0.8
                fractions["classroom_pc"] = 0.8
                fractions["projector"] = 0.8
                fractions["speaker"] = 0.7
        elif is_saturday and not in_vacation:
            # Saturday student presentations / design showcase: 10:00-13:00
            if 10 <= hour < 13:
                scheduled_occupancy = 1
                fractions["fans"] = 0.85
                fractions["lighting"] = 0.85
                fractions["classroom_pc"] = 0.85
                fractions["projector"] = 0.85
                fractions["speaker"] = 0.80

    return fractions, scheduled_occupancy


# ---------------------------------------------------------------------------
# Primary Data Generation Function
# ---------------------------------------------------------------------------

def generate_synthetic_energy_data(
    start_date: str = "2025-01-01 00:00:00",
    end_date: str = "2025-12-31 23:00:00",
    seed: int = DEFAULT_SEED,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generates an hourly synthetic electricity consumption dataset for all 8 monitored departmental spaces
    across the calendar year 2025 using equipment-based calculations.
    
    All readings and schedules are simulated for academic demonstration.
    Ground-truth anomalies are flagged in 'is_anomaly_ground_truth' for academic evaluation only.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    
    rows = []
    
    for space in MONITORED_SPACES:
        room_id = space["room_id"]
        room_name = space["room_name"]
        floor = space["floor"]
        room_type = space["room_type"]
        equipment_list = space["equipment"]

        for ts in timestamps:
            hour = ts.hour
            dow = ts.day_of_week  # 0=Mon, 6=Sun
            month = ts.month
            is_weekend = int(dow in (5, 6))
            in_vacation = is_college_vacation(ts)
            temp_factor = get_seasonal_temperature_factor(month)

            # Determine scheduled activity levels per equipment category
            fractions, scheduled_occupancy = get_equipment_active_fractions(
                room_id=room_id,
                room_type=room_type,
                hour=hour,
                day_of_week=dow,
                in_vacation=in_vacation,
            )

            # Calculate base equipment energy contribution for this hour
            total_equipment_kwh = 0.0
            for eq in equipment_list:
                cat = eq.get("category", "continuous")
                act_frac = fractions.get(cat, 0.0)

                # Add minor realistic human variance during active periods
                if act_frac > 0.05 and cat != "continuous":
                    act_frac = min(1.0, max(0.0, act_frac + rng.uniform(-0.03, 0.03)))

                eq_kwh = compute_hourly_equipment_kwh(eq, act_frac, temp_factor)
                total_equipment_kwh += eq_kwh

            # Add reproducible Gaussian fluctuation (real electrical variations)
            noise = rng.normal(0, 0.015 * total_equipment_kwh + 0.005)
            kwh = max(0.005, total_equipment_kwh + noise)

            # Ground truth educational anomaly injection
            is_anomaly_gt = 0
            anomaly_desc = "Normal Operation"

            # Scenario 1: DSBD Lab - 50 PCs and ventilation left running overnight
            if room_id == "DSBD_LAB" and ts.month == 6 and ts.day in (13, 14) and (hour >= 20 or hour <= 6):
                kwh += rng.uniform(3.5, 4.8)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: 50 PCs and ventilation left running overnight during vacation weekend"

            # Scenario 2: Classroom E501 - Lights and projector left ON overnight
            elif room_id == "CLASS_E501" and ts.month == 10 and ts.day in (15, 16) and (hour >= 21 or hour <= 5):
                kwh += rng.uniform(0.65, 0.85)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: Classroom lights and projector left ON after departmental seminar"

            # Scenario 3: HOD Cabin - Executive PC and fans left running continuously outside office hours
            elif room_id == "HOD_CABIN" and ts.month == 3 and ts.day in (22, 23) and (hour >= 19 or hour <= 7):
                kwh += rng.uniform(0.25, 0.35)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: Executive PC and fans left running continuously outside office hours"

            # Scenario 4: DBMS Lab - Workstations left running full load during Sunday / holiday
            elif room_id == "DBMS_LAB" and ts.month == 9 and ts.day == 21 and 10 <= hour <= 20:
                kwh += rng.uniform(3.8, 5.0)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: High power consumption on Sunday without scheduled session"

            # Scenario 5: DSA Lab - Sudden breaker trip during Monday morning practicals
            elif room_id == "DSA_LAB" and ts.month == 2 and ts.day == 17 and 10 <= hour <= 12:
                kwh = 0.02 + rng.uniform(0.001, 0.005)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: MCB trip / power cut during scheduled lab session"

            # Scenario 6: PBL Lab - Workstations running overnight unattended
            elif room_id == "PBL_LAB" and ts.month == 11 and ts.day == 8 and (hour >= 22 or hour <= 4):
                kwh += rng.uniform(3.0, 4.2)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: Heavy workstation and bench equipment running overnight unattended"

            # Scenario 7: Department Hall - Projector, PA speakers, and 8 fans left ON overnight
            elif room_id == "DEPT_HALL" and ts.month == 12 and ts.day in (5, 6) and (hour >= 20 or hour <= 6):
                kwh += rng.uniform(0.9, 1.3)
                is_anomaly_gt = 1
                anomaly_desc = "Injected: Hall projector, sound system, and 8 fans left ON overnight after event"

            # Scenario 8: Sporadic rare noise anomaly (~0.3% of off-hours records)
            elif rng.random() < 0.003 and hour not in (8, 9, 13, 14):
                if rng.random() < 0.7:
                    kwh += rng.uniform(1.2, 2.5)
                    is_anomaly_gt = 1
                    anomaly_desc = "Injected: Sporadic high surge / equipment left on"
                else:
                    kwh = max(0.005, kwh * 0.15)
                    is_anomaly_gt = 1
                    anomaly_desc = "Injected: Meter drop / partial circuit outage"

            rows.append({
                "timestamp": ts,
                "date": ts.date(),
                "room_id": room_id,
                "room_name": room_name,
                "floor": floor,
                "room_type": room_type,
                "hour": hour,
                "day_of_week": dow,
                "day_name": ts.strftime("%A"),
                "month": month,
                "month_name": ts.strftime("%B"),
                "is_weekend": is_weekend,
                "is_scheduled_occupancy": scheduled_occupancy,
                "kwh": round(float(max(0.0, kwh)), 3),
                "is_anomaly_ground_truth": is_anomaly_gt,
                "ground_truth_note": anomaly_desc,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["timestamp", "room_id"]).reset_index(drop=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DATA GENERATOR] Saved {len(df)} records to {save_path}")

    return df


def load_energy_data(
    csv_path: str | Path = "data/csd_energy_simulated.csv",
    force_recreate: bool = False,
) -> pd.DataFrame:
    """
    Loads dataset from CSV cache if available, or generates it deterministically.
    """
    path = Path(csv_path)
    if path.exists() and not force_recreate:
        df = pd.read_csv(path, parse_dates=["timestamp", "date"])
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # Generate and save
    df = generate_synthetic_energy_data(save_path=path)
    return df


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent
    target_csv = current_dir / "data" / "csd_energy_simulated.csv"
    print(f"Generating synthetic CSD Energy Intelligence dataset at: {target_csv}")
    df_gen = generate_synthetic_energy_data(save_path=target_csv)
    print(f"Dataset generated successfully with {len(df_gen)} rows.")
    print("Room counts:\n", df_gen["room_name"].value_counts())
    print("Ground truth anomalies:\n", df_gen["is_anomaly_ground_truth"].value_counts())
