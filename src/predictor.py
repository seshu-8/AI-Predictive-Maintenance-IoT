"""
predictor.py
------------
Uses the trained model to predict machine failures and generate alerts.

This is the "deployment" module — in a real system, this would run
continuously on live sensor streams from IoT devices.

Alert levels:
  🟢 NORMAL    (failure probability < 30%)  → machine is healthy
  🟡 WARNING   (failure probability 30-70%) → monitor closely
  🔴 CRITICAL  (failure probability > 70%)  → maintenance required NOW
"""

import numpy as np
import pandas as pd
from typing import Union


# ─── ALERT THRESHOLDS ─────────────────────────────────────────────────
NORMAL_THRESHOLD   = 0.30   # below this → no concern
WARNING_THRESHOLD  = 0.70   # above this → critical alert


def predict_failures(model, X_test: np.ndarray) -> np.ndarray:
    """
    Makes binary failure predictions on test data.

    Parameters:
        model:  Trained RandomForest model
        X_test: Feature matrix for test samples

    Returns:
        predictions: Array of 0 (Normal) or 1 (Failure)
    """
    predictions = model.predict(X_test)

    # Count predictions
    failure_count = predictions.sum()
    normal_count  = len(predictions) - failure_count

    print(f"\n  Prediction results:")
    print(f"    🟢 Normal  predictions: {normal_count:>5} ({normal_count/len(predictions)*100:.1f}%)")
    print(f"    🔴 Failure predictions: {failure_count:>5} ({failure_count/len(predictions)*100:.1f}%)")

    return predictions


def predict_with_probability(model, X_test: np.ndarray) -> pd.DataFrame:
    """
    Predicts failure AND returns probability scores.
    Probability is more useful than binary prediction — it shows
    HOW LIKELY a failure is, enabling graduated alerts.

    Parameters:
        model:  Trained RandomForest model
        X_test: Feature matrix

    Returns:
        DataFrame with predictions and probability scores
    """
    # Get class probabilities [prob_normal, prob_failure]
    proba = model.predict_proba(X_test)
    failure_proba = proba[:, 1]  # probability of class=1 (Failure)

    # Binary prediction from model
    binary_pred = model.predict(X_test)

    # Assign alert levels
    alert_levels = []
    for prob in failure_proba:
        if prob < NORMAL_THRESHOLD:
            alert_levels.append("🟢 NORMAL")
        elif prob < WARNING_THRESHOLD:
            alert_levels.append("🟡 WARNING")
        else:
            alert_levels.append("🔴 CRITICAL")

    results_df = pd.DataFrame({
        "predicted_label":     binary_pred,
        "failure_probability": np.round(failure_proba, 4),
        "alert_level":         alert_levels
    })

    return results_df


def generate_alert_report(model, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a full machine-level alert report.
    Groups predictions by machine and reports per-machine risk.
    
    In a real system, this report would be sent to:
    - Maintenance engineers (email/SMS)
    - Dashboard / SCADA system
    - ERP system (SAP, Oracle)
    
    Parameters:
        model:       Trained model
        df_features: Full feature-engineered DataFrame (all machines)

    Returns:
        report_df: Machine-level failure risk summary
    """
    from src.feature_engineer import get_feature_columns

    feature_cols = get_feature_columns(df_features)
    X_all = df_features[feature_cols].values

    # Get probabilities
    proba = model.predict_proba(X_all)
    failure_proba = proba[:, 1]

    df_result = df_features[["machine_id", "cycle", "failure"]].copy()
    df_result["failure_probability"] = np.round(failure_proba, 4)
    df_result["predicted_label"] = model.predict(X_all)

    # Per-machine summary
    machine_summary = df_result.groupby("machine_id").agg(
        avg_failure_prob=("failure_probability", "mean"),
        max_failure_prob=("failure_probability", "max"),
        last_cycle_prob=("failure_probability", "last"),
        total_cycles=("cycle", "max"),
        actual_failures=("failure", "sum")
    ).reset_index()

    # Assign alert level based on LAST CYCLE probability (most recent reading)
    def assign_alert(prob):
        if prob < NORMAL_THRESHOLD:
            return "🟢 NORMAL"
        elif prob < WARNING_THRESHOLD:
            return "🟡 WARNING"
        else:
            return "🔴 CRITICAL"

    machine_summary["alert_level"] = machine_summary["last_cycle_prob"].apply(assign_alert)

    # Sort by risk (highest first)
    machine_summary = machine_summary.sort_values("last_cycle_prob", ascending=False)

    print("\n  ─── MACHINE FAILURE ALERT REPORT ───────────────────────")
    print(machine_summary[["machine_id", "last_cycle_prob", "max_failure_prob",
                            "total_cycles", "alert_level"]].to_string(index=False))
    print("  ─────────────────────────────────────────────────────────")

    # Save report
    machine_summary.to_csv("outputs/alert_report.csv", index=False)
    print("  ✓ Alert report saved to outputs/alert_report.csv")

    return machine_summary


# ─── QUICK TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from src.data_loader import load_data
    from src.preprocessor import preprocess_data
    from src.feature_engineer import engineer_features
    from src.model import train_model, evaluate_model

    df_raw = load_data(save_csv=False)
    df_clean = preprocess_data(df_raw)
    df_features = engineer_features(df_clean)
    model, X_test, y_test, feature_cols = train_model(df_features)

    print("\nRunning prediction with probabilities (first 10 samples):")
    results = predict_with_probability(model, X_test[:10])
    print(results)

    import os
    os.makedirs("outputs", exist_ok=True)
    report = generate_alert_report(model, df_features)
