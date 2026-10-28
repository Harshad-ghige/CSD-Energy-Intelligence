"""
Unit Tests for CSD Energy Intelligence
Tests calculations, preprocessing, ML training, anomaly detection, and savings simulations.
Can be executed via `python -m unittest discover tests`.
"""

import unittest
import numpy as np
import pandas as pd

from src.data_generator import MONITORED_SPACES, generate_synthetic_energy_data
from src.preprocessing import (
    filter_dataset,
    create_ml_features,
    prepare_clustering_room_features,
)
from src.analytics import (
    calculate_total_energy,
    calculate_daily_average,
    calculate_peak_consumption,
    calculate_deviation,
    calculate_percentage_deviation,
    calculate_summary_metrics,
    compute_hourly_profile,
    compute_room_breakdown,
    compute_floor_comparison,
)
from src.anomaly_detection import run_isolation_forest_detection, evaluate_anomaly_detector
from src.prediction import train_room_energy_model, generate_seven_day_forecast
from src.clustering import run_room_clustering
from src.savings import calculate_savings_scenario, compare_multiple_scenarios


class TestCSDEnergyIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate a small 14-day sample dataset for fast testing
        cls.df = generate_synthetic_energy_data(
            start_date="2025-03-01 00:00:00",
            end_date="2025-03-14 23:00:00",
            seed=42,
        )

    def test_monitored_spaces_specification(self):
        """Ensure all 8 designated departmental spaces are present."""
        self.assertEqual(len(MONITORED_SPACES), 8)
        room_ids = [s["room_id"] for s in MONITORED_SPACES]
        expected_ids = ["DSBD_LAB", "DBMS_LAB", "CLASS_E501", "CLASS_E502", "DSA_LAB", "PBL_LAB", "HOD_CABIN", "DEPT_HALL"]
        for eid in expected_ids:
            self.assertIn(eid, room_ids)

        # Floor distribution check
        third_floor = [s for s in MONITORED_SPACES if s["floor"] == "3rd Floor"]
        fourth_floor = [s for s in MONITORED_SPACES if s["floor"] == "4th Floor"]
        self.assertEqual(len(third_floor), 2)
        self.assertEqual(len(fourth_floor), 6)

    def test_synthetic_data_integrity(self):
        """Verify synthetic data columns, types, and values."""
        self.assertFalse(self.df.empty)
        required_cols = [
            "timestamp", "date", "room_id", "room_name", "floor",
            "room_type", "hour", "day_of_week", "month", "is_weekend",
            "kwh", "is_anomaly_ground_truth"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)

        # Check positive energy readings
        self.assertTrue((self.df["kwh"] >= 0).all())

        # Check 8 unique rooms in output
        self.assertEqual(self.df["room_name"].nunique(), 8)

    def test_calculation_functions(self):
        """Verify core energy and deviation calculation accuracy."""
        test_series = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=5, freq="h"),
            "kwh": [10.0, 20.0, 30.0, 40.0, 50.0],
            "room_name": ["DSBD Lab"] * 5
        })

        # Total
        self.assertEqual(calculate_total_energy(test_series), 150.0)

        # Peak
        peak_val, peak_ts, peak_room = calculate_peak_consumption(test_series)
        self.assertEqual(peak_val, 50.0)
        self.assertEqual(peak_room, "DSBD Lab")

        # Deviations
        self.assertAlmostEqual(calculate_deviation(12.5, 10.0), 2.5)
        self.assertAlmostEqual(calculate_deviation(8.0, 10.0), -2.0)

        # Percentage deviation
        self.assertAlmostEqual(calculate_percentage_deviation(12.5, 10.0), 25.0)
        self.assertAlmostEqual(calculate_percentage_deviation(8.0, 10.0), -20.0)

        # Percentage deviation zero-handling
        self.assertEqual(calculate_percentage_deviation(0.0, 0.0), 0.0)
        self.assertEqual(calculate_percentage_deviation(5.0, 0.0), 100.0)

    def test_empty_dataset_handling(self):
        """Verify edge case handling for empty or null dataframes."""
        empty_df = pd.DataFrame()
        self.assertEqual(calculate_total_energy(empty_df), 0.0)
        self.assertEqual(calculate_daily_average(empty_df), 0.0)
        metrics = calculate_summary_metrics(empty_df)
        self.assertEqual(metrics["total_kwh"], 0.0)
        self.assertEqual(metrics["monitored_spaces_count"], 0)

    def test_preprocessing_filters_and_features(self):
        """Test dataset filtering and ML feature generation."""
        filtered = filter_dataset(
            self.df,
            start_date="2025-03-02",
            end_date="2025-03-05",
            selected_rooms=["DSBD Lab"],
        )
        self.assertEqual(filtered["room_name"].nunique(), 1)
        self.assertEqual(filtered["room_name"].iloc[0], "DSBD Lab")

        # Feature creation
        ml_df = create_ml_features(filtered, include_lags=True)
        self.assertIn("hour_sin", ml_df.columns)
        self.assertIn("lag_24h", ml_df.columns)
        self.assertIn("rolling_mean_24h", ml_df.columns)
        self.assertFalse(ml_df["lag_24h"].isna().any())

    def test_anomaly_detection_model(self):
        """Verify Isolation Forest training and detection functionality."""
        detected_df, metrics = run_isolation_forest_detection(self.df, contamination=0.03, random_state=42)
        self.assertIn("is_anomaly_detected", detected_df.columns)
        self.assertIn("deviation_kwh", detected_df.columns)
        self.assertIn("anomaly_direction", detected_df.columns)
        self.assertTrue(metrics["total_anomalies_detected"] > 0)

    def test_prediction_model(self):
        """Verify RandomForestRegressor training with strict chronological split."""
        ml_df = create_ml_features(self.df, include_lags=True)
        model, metrics, test_df, feat_imp = train_room_energy_model(
            ml_df,
            room_id="DSBD_LAB",
            train_split_ratio=0.8,
            n_estimators=30,
            random_state=42,
        )
        self.assertIsNotNone(model)
        self.assertGreater(metrics["mae"], 0.0)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertLess(metrics["train_end"], metrics["test_start"])
        self.assertFalse(feat_imp.empty)

        # Test 7-day forecast
        forecast_df = generate_seven_day_forecast(model, ml_df, room_id="DSBD_LAB", forecast_days=7)
        self.assertEqual(len(forecast_df), 7 * 24)
        self.assertTrue((forecast_df["forecast_kwh"] > 0).all())

    def test_clustering_kmeans_pca(self):
        """Verify KMeans behavioral clustering and PCA projection."""
        room_feats = prepare_clustering_room_features(self.df)
        self.assertEqual(len(room_feats), 8)
        clustered_df, summary_df, meta = run_room_clustering(room_feats, n_clusters=3, random_state=42)
        self.assertIn("cluster", clustered_df.columns)
        self.assertIn("pca_x", clustered_df.columns)
        self.assertIn("pca_y", clustered_df.columns)
        self.assertEqual(len(clustered_df), 8)
        self.assertFalse(summary_df.empty)

    def test_savings_simulator(self):
        """Verify hypothetical what-if savings calculation logic."""
        res = calculate_savings_scenario(baseline_kwh=1000.0, reduction_percentage=10.0, tariff_inr_per_kwh=8.0)
        self.assertEqual(res["saved_kwh"], 100.0)
        self.assertEqual(res["projected_kwh"], 900.0)
        self.assertEqual(res["saved_cost_inr"], 800.0)
        self.assertEqual(res["projected_cost_inr"], 7200.0)

        # Multi-scenario table
        multi = compare_multiple_scenarios(
            baseline_kwh=1000.0,
            reduction_percentages=[5.0, 10.0, 15.0],
            tariff_inr_per_kwh=8.0,
        )
        self.assertEqual(len(multi), 3)


if __name__ == "__main__":
    unittest.main()
