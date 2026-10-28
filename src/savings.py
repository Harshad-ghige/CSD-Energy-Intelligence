"""
CSD Energy Intelligence - Energy Savings Simulator
Calculates hypothetical what-if reduction scenarios and estimated cost savings in INR (₹).
Does not count unverified statistical anomalies as confirmed savings.
Avoids unverified ecological claims.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_savings_scenario(
    baseline_kwh: float,
    reduction_percentage: float,
    tariff_inr_per_kwh: float,
    period_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes hypothetical savings for a single scenario.
    """
    reduction_pct = max(0.0, min(100.0, float(reduction_percentage)))
    tariff = max(0.0, float(tariff_inr_per_kwh))
    baseline = max(0.0, float(baseline_kwh))

    saved_kwh = baseline * (reduction_pct / 100.0)
    projected_kwh = baseline - saved_kwh
    saved_cost_inr = saved_kwh * tariff
    original_cost_inr = baseline * tariff
    projected_cost_inr = projected_kwh * tariff

    # Annualized projections if period_days is known
    annual_saved_kwh = None
    annual_saved_inr = None
    if period_days and period_days > 0:
        annual_factor = 365.0 / period_days
        annual_saved_kwh = round(saved_kwh * annual_factor, 2)
        annual_saved_inr = round(saved_cost_inr * annual_factor, 2)

    return {
        "baseline_kwh": round(baseline, 2),
        "reduction_percentage": round(reduction_pct, 1),
        "saved_kwh": round(saved_kwh, 2),
        "projected_kwh": round(projected_kwh, 2),
        "tariff_inr_per_kwh": round(tariff, 2),
        "saved_cost_inr": round(saved_cost_inr, 2),
        "original_cost_inr": round(original_cost_inr, 2),
        "projected_cost_inr": round(projected_cost_inr, 2),
        "annual_saved_kwh": annual_saved_kwh,
        "annual_saved_inr": annual_saved_inr,
    }


def compare_multiple_scenarios(
    baseline_kwh: float,
    reduction_percentages: List[float],
    tariff_inr_per_kwh: float,
    period_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a comparative DataFrame across multiple target reduction percentages (e.g., 5%, 10%, 15%, 20%).
    """
    rows = []
    for pct in reduction_percentages:
        res = calculate_savings_scenario(baseline_kwh, pct, tariff_inr_per_kwh, period_days)
        rows.append({
            "Scenario Reduction (%)": f"{res['reduction_percentage']}%",
            "Saved Energy (kWh)": res["saved_kwh"],
            "Projected Consumption (kWh)": res["projected_kwh"],
            "Estimated Savings (₹)": f"₹{res['saved_cost_inr']:,.2f}",
            "Projected Cost (₹)": f"₹{res['projected_cost_inr']:,.2f}",
            "Annualized Savings (kWh)": res["annual_saved_kwh"] if res["annual_saved_kwh"] is not None else "N/A",
            "Annualized Savings (₹)": f"₹{res['annual_saved_inr']:,.2f}" if res["annual_saved_inr"] is not None else "N/A",
        })
    return pd.DataFrame(rows)
