"""
CSD Energy Intelligence - Data Preprocessing & Feature Engineering
Provides robust preprocessing, filtering, and feature extraction functions for analytics and ML models.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def filter_dataset(
    df: pd.DataFrame,
    start_date: Optional[Union[str, pd.Timestamp, object]] = None,
    end_date: Optional[Union[str, pd.Timestamp, object]] = None,
    selected_rooms: Optional[List[str]] = None,
    selected_floors: Optional[List[str]] = None,
    selected_room_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filters the dataset by date range, selected rooms, floors, and room types safely.
    Handles None and empty selections gracefully.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(filtered["timestamp"]):
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])

    # Date range filtering
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        filtered = filtered[filtered["timestamp"] >= start_ts]

    if end_date is not None:
        # Include the full end_date day (up to 23:59:59)
        end_ts = pd.to_datetime(end_date)
        if hasattr(end_date, "hour") and end_date.hour == 0 and hasattr(end_date, "minute") and end_date.minute == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[filtered["timestamp"] <= end_ts]

    # Room filter
    if selected_rooms:
        filtered = filtered[filtered["room_name"].isin(selected_rooms)]

    # Floor filter
    if selected_floors:
        filtered = filtered[filtered["floor"].isin(selected_floors)]

    # Room type filter
    if selected_room_types:
        filtered = filtered[filtered["room_type"].isin(selected_room_types)]

    return filtered.reset_index(drop=True)


def create_ml_features(df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """
    Generates temporal, cyclical, and historical lag features for ML models.
    Operates per room to avoid cross-room leakage.
    Never includes ground truth anomaly flags in feature sets.
    """
    if df.empty:
        return df

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Ensure chronological order per room
    data = data.sort_values(by=["room_id", "timestamp"]).reset_index(drop=True)

    # Cyclical hour encoding (24 hours circle)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)

    # Cyclical day of week encoding (7 days circle)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    # Cyclical month encoding (12 months circle)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Binary flags
    data["is_academic_hours"] = ((data["hour"] >= 8) & (data["hour"] <= 18) & (data["is_weekend"] == 0)).astype(int)

    if include_lags:
        # Group by room_id to ensure lag values are room-specific
        grouped = data.groupby("room_id")["kwh"]
        data["lag_1h"] = grouped.shift(1)
        data["lag_24h"] = grouped.shift(24)       # same hour yesterday
        data["lag_168h"] = grouped.shift(168)     # same hour last week
        data["rolling_mean_24h"] = grouped.transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())

        # Forward fill or back fill any initial NaN values caused by shifts
        data["lag_1h"] = data["lag_1h"].bfill().fillna(data["kwh"])
        data["lag_24h"] = data["lag_24h"].bfill().fillna(data["kwh"])
        data["lag_168h"] = data["lag_168h"].bfill().fillna(data["kwh"])
        data["rolling_mean_24h"] = data["rolling_mean_24h"].bfill().fillna(data["kwh"])

    # Re-sort chronologically by timestamp and room
    data = data.sort_values(by=["timestamp", "room_id"]).reset_index(drop=True)
    return data


def prepare_clustering_room_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts behavioral consumption features per room for KMeans clustering:
    - Average hourly kWh
    - Peak hourly kWh
    - Median hourly kWh
    - Nighttime consumption ratio (22:00 - 06:00 vs total)
    - Weekend consumption ratio (Sat/Sun vs total)
    - Working hours mean kWh (09:00 - 17:00 weekdays)
    - Peak to average ratio
    """
    if df.empty:
        return pd.DataFrame()

    features_list = []
    
    for room_name, group in df.groupby("room_name"):
        room_id = group["room_id"].iloc[0]
        floor = group["floor"].iloc[0]
        room_type = group["room_type"].iloc[0]
        
        total_kwh = group["kwh"].sum()
        mean_kwh = group["kwh"].mean()
        peak_kwh = group["kwh"].max()
        median_kwh = group["kwh"].median()
        std_kwh = group["kwh"].std()

        # Nighttime ratio (hours 22 to 23 and 0 to 6)
        night_mask = (group["hour"] >= 22) | (group["hour"] <= 6)
        night_kwh = group.loc[night_mask, "kwh"].sum()
        night_ratio = night_kwh / total_kwh if total_kwh > 0 else 0.0

        # Weekend ratio
        weekend_mask = group["is_weekend"] == 1
        weekend_kwh = group.loc[weekend_mask, "kwh"].sum()
        weekend_ratio = weekend_kwh / total_kwh if total_kwh > 0 else 0.0

        # Working hours mean (09:00 - 17:00 on weekdays)
        working_mask = (group["hour"] >= 9) & (group["hour"] <= 17) & (group["is_weekend"] == 0)
        working_mean = group.loc[working_mask, "kwh"].mean() if working_mask.any() else mean_kwh

        # Peak to average ratio
        peak_to_avg = peak_kwh / mean_kwh if mean_kwh > 0 else 1.0

        features_list.append({
            "room_name": room_name,
            "room_id": room_id,
            "floor": floor,
            "room_type": room_type,
            "mean_kwh": round(float(mean_kwh), 3),
            "peak_kwh": round(float(peak_kwh), 3),
            "median_kwh": round(float(median_kwh), 3),
            "std_kwh": round(float(std_kwh), 3),
            "night_ratio": round(float(night_ratio), 4),
            "weekend_ratio": round(float(weekend_ratio), 4),
            "working_mean_kwh": round(float(working_mean), 3),
            "peak_to_avg_ratio": round(float(peak_to_avg), 3),
            "total_kwh": round(float(total_kwh), 1),
        })

    return pd.DataFrame(features_list)
