"""
main.py
-------
AI-Powered Predictive Maintenance System for IoT Devices
=========================================================
Author  : Your Name
Dataset : Simulated IoT Sensor Data (10 virtual machines)
Model   : Random Forest Classifier
Task    : Binary classification — predict machine failure
GitHub  : https://github.com/YOUR_USERNAME/AI-Predictive-Maintenance-IoT

HOW TO RUN:
    python main.py

Outputs:
    - data/raw/sensor_data.csv          ← raw generated data
    - data/processed/clean_data.csv     ← after preprocessing
    - data/processed/engineered_features.csv ← all features
    - models/predictive_model.pkl       ← trained model
    - models/scaler.pkl                 ← saved scaler
    - outputs/alert_report.csv          ← machine risk report
    - outputs/01_sensor_readings.png
    - outputs/02_failure_distribution.png
    - outputs/03_confusion_matrix.png
    - outputs/04_feature_importance.png
    - outputs/05_actual_vs_predicted.png
    - outputs/06_failure_probability_timeline.png
"""

import os
import sys
import time

# ── Ensure src/ is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.feature_engineer import engineer_features, get_feature_columns
from src.model import train_model, evaluate_model, save_model
from src.predictor import predict_failures, predict_with_probability, generate_alert_report
from src.visualizer import (
    plot_sensor_data,
    plot_failure_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_predictions,
    plot_failure_probability_timeline
)

DIVIDER = "=" * 65


def banner():
    print(DIVIDER)
    print("  AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES")
    print(DIVIDER)
    print("  Simulating: 10 industrial machines with 6 sensor types")
    print("  Model:      Random Forest Classifier")
    print("  Objective:  Predict machine failure 30 cycles in advance")
    print(DIVIDER)


def phase(num: int, name: str):
    print(f"\n{'─' * 65}")
    print(f"  PHASE {num}: {name.upper()}")
    print(f"{'─' * 65}")


def main():
    start_time = time.time()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    banner()

    # ─── PHASE 1: LOAD DATA ───────────────────────────────────────────
    phase(1, "Data Loading")
    df_raw = load_data(save_csv=True)

    # ─── PHASE 2: PREPROCESSING ───────────────────────────────────────
    phase(2, "Data Preprocessing")
    df_clean = preprocess_data(df_raw)

    # ─── PHASE 3: FEATURE ENGINEERING ────────────────────────────────
    phase(3, "Feature Engineering")
    df_features = engineer_features(df_clean)
    feature_cols = get_feature_columns(df_features)

    # ─── PHASE 4: MODEL TRAINING ──────────────────────────────────────
    phase(4, "Model Training (Random Forest)")
    model, X_test, y_test, feature_cols = train_model(df_features)

    # ─── PHASE 5: MODEL EVALUATION ───────────────────────────────────
    phase(5, "Model Evaluation")
    metrics = evaluate_model(model, X_test, y_test)

    # ─── PHASE 6: SAVE MODEL ──────────────────────────────────────────
    phase(6, "Saving Model")
    save_model(model, "models/predictive_model.pkl")

    # ─── PHASE 7: PREDICTIONS ────────────────────────────────────────
    phase(7, "Failure Predictions")
    predictions = predict_failures(model, X_test)

    print("\n  Sample predictions with probabilities (first 10):")
    sample_results = predict_with_probability(model, X_test[:10])
    print(sample_results.to_string(index=False))

    # Machine-level alert report
    report = generate_alert_report(model, df_features)

    # ─── PHASE 8: VISUALIZATIONS ─────────────────────────────────────
    phase(8, "Generating Visualizations")
    print("  Generating all charts... (saved to outputs/ folder)")
    plot_sensor_data(df_clean)
    plot_failure_distribution(df_features)
    plot_confusion_matrix(y_test, predictions)
    plot_feature_importance(model, feature_cols)
    plot_predictions(y_test, predictions)
    plot_failure_probability_timeline(model, df_features)

    # ─── FINAL SUMMARY ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{DIVIDER}")
    print("  PROJECT COMPLETE!")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(DIVIDER)
    print("\n  📊 MODEL PERFORMANCE SUMMARY:")
    print(f"    Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"    Precision : {metrics['precision']*100:.2f}%")
    print(f"    Recall    : {metrics['recall']*100:.2f}%")
    print(f"    F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"\n  📁 OUTPUT FILES:")
    for f in sorted(os.listdir("outputs")):
        print(f"    outputs/{f}")
    print(f"    models/predictive_model.pkl")
    print(f"    models/scaler.pkl")
    print(DIVIDER)
    print("\n  ✅ Push all files to GitHub to complete your proof of work!")
    print(f"  ✅ Add screenshots from outputs/ folder to your README")
    print(DIVIDER)


if __name__ == "__main__":
    main()
