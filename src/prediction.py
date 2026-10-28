"""
CSD Energy Intelligence - Consumption Prediction & Forecasting
Implements RandomForestRegressor with strict chronological train/test splitting to prevent data leakage.
Computes real test-set metrics (MAE, RMSE, R²) and generates 7-day future forecasts.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_scheduled_occupancy",
    "is_academic_hours",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
]


def train_room_energy_model(
    df: pd.DataFrame,
    room_id: Optional[str] = None,
    train_split_ratio: float = 0.80,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Trains a RandomForestRegressor on chronological historical data.
    Ensures zero data leakage: train set comes strictly before test set chronologically.
    Returns: (fitted_model, metrics_dict, test_df_with_predictions, feature_importance_df)
    """
    if df is None or df.empty:
        raise ValueError("Cannot train model on empty DataFrame.")

    # Filter room if specified
    data = df.copy()
    if room_id:
        data = data[data["room_id"] == room_id].copy()

    # Ensure chronological sorting
    data = data.sort_values(by="timestamp").reset_index(drop=True)

    # Check for feature columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for prediction: {missing_cols}")

    # Chronological Split
    split_index = int(len(data) * train_split_ratio)
    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["kwh"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["kwh"]

    # Model training
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    test_df["predicted_kwh"] = np.round(y_pred, 3)
    test_df["error_kwh"] = np.round(test_df["kwh"] - test_df["predicted_kwh"], 3)
    test_df["abs_error_kwh"] = np.abs(test_df["error_kwh"])

    # Metrics calculation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
    }

    # Feature Importance
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": np.round(model.feature_importances_, 4),
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return model, metrics, test_df, feat_imp


def generate_seven_day_forecast(
    model: RandomForestRegressor,
    history_df: pd.DataFrame,
    room_id: str,
    forecast_days: int = 7,
) -> pd.DataFrame:
    """
    Generates a forward-looking hourly forecast for the next 7 days for the specified room.
    Constructs future calendar features and autoregressive rolling values.
    """
    if history_df.empty:
        return pd.DataFrame()

    room_data = history_df[history_df["room_id"] == room_id].sort_values(by="timestamp").reset_index(drop=True)
    if room_data.empty:
        return pd.DataFrame()

    last_ts = room_data["timestamp"].max()
    future_timestamps = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=forecast_days * 24, freq="h")

    # Seed rolling state with last 168 hours of actual history
    recent_kwh_history = list(room_data["kwh"].tail(168).values)
    room_name = room_data["room_name"].iloc[0]
    floor = room_data["floor"].iloc[0]
    room_type = room_data["room_type"].iloc[0]

    forecast_rows = []

    for ts in future_timestamps:
        h = ts.hour
        dow = ts.day_of_week
        m = ts.month
        is_wknd = int(dow in (5, 6))
        is_acad = int((8 <= h <= 18) and not is_wknd)
        
        # Simulated scheduled occupancy based on timetable
        sched_occ = 1 if (is_acad and ((9 <= h < 13) or (14 <= h < 17))) else 0

        # Lags from autoregressive state
        lag_24 = recent_kwh_history[-24] if len(recent_kwh_history) >= 24 else np.mean(recent_kwh_history)
        lag_168 = recent_kwh_history[-168] if len(recent_kwh_history) >= 168 else np.mean(recent_kwh_history)
        rolling_24 = np.mean(recent_kwh_history[-24:]) if len(recent_kwh_history) >= 24 else np.mean(recent_kwh_history)

        feature_vector = pd.DataFrame([{
            "hour": h,
            "day_of_week": dow,
            "month": m,
            "is_weekend": iswknd if 'iswknd' in locals() else is_wknd,
            "is_scheduled_occupancy": sched_occ,
            "is_academic_hours": is_acad,
            "hour_sin": np.sin(2 * np.pi * h / 24.0),
            "hour_cos": np.cos(2 * np.pi * h / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "lag_24h": lag_24,
            "lag_168h": lag_168,
            "rolling_mean_24h": rolling_24,
        }])[FEATURE_COLUMNS]

        pred_val = float(model.predict(feature_vector)[0])
        pred_val = max(0.1, pred_val)

        # Update running history for recursive lags
        recent_kwh_history.append(pred_val)

        forecast_rows.append({
            "timestamp": ts,
            "room_id": room_id,
            "room_name": room_name,
            "floor": floor,
            "room_type": room_type,
            "hour": h,
            "day_name": ts.strftime("%A"),
            "forecast_kwh": round(pred_val, 3),
            "lower_bound_kwh": round(max(0.05, pred_val * 0.88), 3),
            "upper_bound_kwh": round(pred_val * 1.12, 3),
        })

    return pd.DataFrame(forecast_rows)
