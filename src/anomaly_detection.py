"""
CSD Energy Intelligence - Anomaly Detection Engine
Implements unsupervised anomaly detection using Scikit-learn IsolationForest.
Strictly does NOT use ground-truth anomaly labels for training.
Calculates historical baselines, deviations, and evaluation metrics for academic demonstration.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_room_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes expected baseline consumption for each (room_id, hour, is_weekend) combination.
    Uses median to ensure robustness against existing anomalies.
    """
    baseline = (
        df.groupby(["room_id", "hour", "is_weekend"])["kwh"]
        .agg(
            expected_median_kwh="median",
            expected_mean_kwh="mean",
            expected_std_kwh="std",
            q25_kwh=lambda x: x.quantile(0.25),
            q75_kwh=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    baseline["expected_std_kwh"] = baseline["expected_std_kwh"].fillna(0.1)
    return baseline


def run_isolation_forest_detection(
    df: pd.DataFrame,
    contamination: float = 0.015,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains an Isolation Forest model per room to detect statistically unusual consumption patterns.
    Uses only unsupervised features: [kwh, hour, day_of_week, is_weekend].
    Never includes ground truth anomaly flags in training.
    """
    if df is None or df.empty:
        return pd.DataFrame(), {}

    # Merge expected baselines
    baselines = compute_room_baselines(df)
    merged = df.merge(baselines, on=["room_id", "hour", "is_weekend"], how="left")

    results = []
    models_summary = {}

    for room_id, room_df in merged.groupby("room_id"):
        room_name = room_df["room_name"].iloc[0]
        # Features for unsupervised Isolation Forest
        feature_cols = ["kwh", "hour", "day_of_week", "is_weekend"]
        X = room_df[feature_cols].copy()

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        # raw preds: -1 for outlier, 1 for inlier
        raw_preds = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)

        room_res = room_df.copy()
        room_res["anomaly_score"] = np.round(anomaly_scores, 4)
        # Convert to 1 = Anomaly, 0 = Normal
        room_res["is_anomaly_detected"] = (raw_preds == -1).astype(int)

        # Baseline deviation calculations
        expected = room_res["expected_median_kwh"]
        room_res["expected_kwh"] = expected.round(3)
        room_res["deviation_kwh"] = (room_res["kwh"] - expected).round(3)
        
        # Percentage deviation
        room_res["deviation_pct"] = np.where(
            expected > 0.01,
            ((room_res["kwh"] - expected) / expected) * 100.0,
            np.where(room_res["kwh"] > 0.01, 100.0, 0.0)
        ).round(2)

        # Categorize anomaly type
        room_res["anomaly_direction"] = np.where(
            room_res["is_anomaly_detected"] == 1,
            np.where(room_res["deviation_kwh"] > 0, "High Consumption Surge", "Abnormally Low Drop"),
            "Normal"
        )

        results.append(room_res)

        models_summary[room_id] = {
            "room_name": room_name,
            "detected_count": int((raw_preds == -1).sum()),
            "total_records": len(room_df),
        }

    detected_df = pd.concat(results, ignore_index=True)
    detected_df = detected_df.sort_values(by=["timestamp", "room_id"]).reset_index(drop=True)

    # Compute DSBD evaluation metrics comparing model predictions with simulated ground truth
    eval_metrics = evaluate_anomaly_detector(detected_df)

    return detected_df, eval_metrics


def evaluate_anomaly_detector(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares IsolationForest detections with simulated ground-truth anomalies.
    Provides Precision, Recall, F1-score, and confusion matrix for academic viva evaluation.
    """
    if "is_anomaly_ground_truth" not in df.columns or "is_anomaly_detected" not in df.columns:
        return {}

    y_true = df["is_anomaly_ground_truth"].values
    y_pred = df["is_anomaly_detected"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "total_anomalies_detected": int(y_pred.sum()),
        "total_ground_truth_anomalies": int(y_true.sum()),
    }
