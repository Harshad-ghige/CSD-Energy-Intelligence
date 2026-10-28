"""
CSD Energy Intelligence - Core Analytics & Reusable Calculations
Contains verified calculation functions for consumption metrics, deviations, profiles, and floor breakdowns.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_total_energy(df: pd.DataFrame) -> float:
    """Calculates sum of kWh readings over the dataframe."""
    if df is None or df.empty or "kwh" not in df.columns:
        return 0.0
    return float(df["kwh"].sum())


def calculate_daily_average(df: pd.DataFrame) -> float:
    """Calculates average daily energy consumption in kWh."""
    if df is None or df.empty or "kwh" not in df.columns:
        return 0.0
    daily_sums = df.groupby(df["timestamp"].dt.date)["kwh"].sum()
    if daily_sums.empty:
        return 0.0
    return float(daily_sums.mean())


def calculate_peak_consumption(df: pd.DataFrame) -> Tuple[float, Optional[pd.Timestamp], Optional[str]]:
    """
    Returns (peak_kwh, peak_timestamp, peak_room_name).
    """
    if df is None or df.empty or "kwh" not in df.columns:
        return 0.0, None, None
    peak_idx = df["kwh"].idxmax()
    peak_row = df.loc[peak_idx]
    return float(peak_row["kwh"]), peak_row["timestamp"], peak_row.get("room_name")


def calculate_deviation(actual: float, expected: float) -> float:
    """Calculates actual - expected deviation in kWh."""
    return float(actual - expected)


def calculate_percentage_deviation(actual: float, expected: float, epsilon: float = 1e-6) -> float:
    """
    Calculates percentage deviation safely: ((actual - expected) / max(abs(expected), epsilon)) * 100.
    Handles zero or negative expected values without division-by-zero errors.
    """
    denom = abs(expected)
    if denom < epsilon:
        if abs(actual) < epsilon:
            return 0.0
        return 100.0 if actual > 0 else -100.0
    return float(((actual - expected) / denom) * 100.0)


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive summary metrics for dashboard cards."""
    if df is None or df.empty:
        return {
            "total_kwh": 0.0,
            "daily_avg_kwh": 0.0,
            "peak_kwh": 0.0,
            "peak_room": "N/A",
            "peak_timestamp": "N/A",
            "monitored_spaces_count": 0,
            "total_hours": 0,
            "lab_total_kwh": 0.0,
            "classroom_total_kwh": 0.0,
            "hod_total_kwh": 0.0,
            "floor_3_kwh": 0.0,
            "floor_4_kwh": 0.0,
        }

    total_kwh = calculate_total_energy(df)
    daily_avg_kwh = calculate_daily_average(df)
    peak_kwh, peak_ts, peak_room = calculate_peak_consumption(df)
    monitored_spaces = df["room_name"].nunique()
    total_hours = df["timestamp"].nunique()

    # Room type splits
    type_groups = df.groupby("room_type")["kwh"].sum().to_dict()
    lab_total = type_groups.get("Laboratory", 0.0)
    classroom_total = type_groups.get("Classroom", 0.0)
    hod_total = type_groups.get("HOD Cabin / Office", 0.0)

    # Floor splits
    floor_groups = df.groupby("floor")["kwh"].sum().to_dict()
    floor_3 = floor_groups.get("3rd Floor", 0.0)
    floor_4 = floor_groups.get("4th Floor", 0.0)

    return {
        "total_kwh": round(total_kwh, 2),
        "daily_avg_kwh": round(daily_avg_kwh, 2),
        "peak_kwh": round(peak_kwh, 2),
        "peak_room": peak_room or "N/A",
        "peak_timestamp": str(peak_ts) if peak_ts else "N/A",
        "monitored_spaces_count": monitored_spaces,
        "total_hours": total_hours,
        "lab_total_kwh": round(lab_total, 2),
        "classroom_total_kwh": round(classroom_total, 2),
        "hod_total_kwh": round(hod_total, 2),
        "floor_3_kwh": round(floor_3, 2),
        "floor_4_kwh": round(floor_4, 2),
    }


def compute_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Computes average consumption by hour, split into Weekday vs Weekend."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["hour", "day_category", "avg_kwh", "min_kwh", "max_kwh"])

    temp = df.copy()
    temp["day_category"] = temp["is_weekend"].map({0: "Weekday (Mon-Fri)", 1: "Weekend (Sat-Sun)"})
    
    profile = (
        temp.groupby(["hour", "day_category"])["kwh"]
        .agg(avg_kwh="mean", min_kwh="min", max_kwh="max")
        .reset_index()
    )
    profile["avg_kwh"] = profile["avg_kwh"].round(3)
    profile["min_kwh"] = profile["min_kwh"].round(3)
    profile["max_kwh"] = profile["max_kwh"].round(3)
    return profile


def compute_room_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Computes room-level summary statistics."""
    if df is None or df.empty:
        return pd.DataFrame()

    agg_df = (
        df.groupby(["room_name", "floor", "room_type"])["kwh"]
        .agg(
            total_kwh="sum",
            mean_kwh="mean",
            peak_kwh="max",
            min_kwh="min",
            std_kwh="std",
        )
        .reset_index()
    )
    total_all = agg_df["total_kwh"].sum()
    agg_df["share_pct"] = ((agg_df["total_kwh"] / total_all) * 100.0).round(2) if total_all > 0 else 0.0
    agg_df["total_kwh"] = agg_df["total_kwh"].round(2)
    agg_df["mean_kwh"] = agg_df["mean_kwh"].round(3)
    agg_df["peak_kwh"] = agg_df["peak_kwh"].round(3)
    agg_df["min_kwh"] = agg_df["min_kwh"].round(3)
    agg_df["std_kwh"] = agg_df["std_kwh"].round(3)
    return agg_df.sort_values(by="total_kwh", ascending=False).reset_index(drop=True)


def compute_floor_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comparative metrics between 3rd Floor and 4th Floor."""
    if df is None or df.empty:
        return pd.DataFrame()

    floor_agg = (
        df.groupby("floor")["kwh"]
        .agg(
            total_kwh="sum",
            mean_hourly_kwh="mean",
            peak_kwh="max",
            rooms_monitored=lambda x: df.loc[x.index, "room_name"].nunique(),
        )
        .reset_index()
    )
    total_all = floor_agg["total_kwh"].sum()
    floor_agg["share_pct"] = ((floor_agg["total_kwh"] / total_all) * 100.0).round(2) if total_all > 0 else 0.0
    floor_agg["total_kwh"] = floor_agg["total_kwh"].round(2)
    floor_agg["mean_hourly_kwh"] = floor_agg["mean_hourly_kwh"].round(3)
    floor_agg["peak_kwh"] = floor_agg["peak_kwh"].round(3)
    return floor_agg
