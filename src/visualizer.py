"""
visualizer.py
-------------
Generates all charts and visualizations for the project.

Charts produced:
  1. Sensor readings over time (multi-machine comparison)
  2. Failure class distribution (bar chart)
  3. Confusion matrix (model evaluation)
  4. Feature importance (top 20 most predictive features)
  5. Failure probability timeline per machine
  6. Sensor degradation curve (before vs after failure zone)

All outputs saved to the outputs/ folder.
These are your GitHub proof screenshots!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─── STYLE SETUP ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_palette("husl")
os.makedirs("outputs", exist_ok=True)


def plot_sensor_data(df: pd.DataFrame, machine_ids: list = None):
    """
    Plots sensor readings over time for selected machines.
    Shows how each sensor changes as the machine approaches failure.
    Failure zone is highlighted in red.
    """
    if machine_ids is None:
        machine_ids = df["machine_id"].unique()[:3]  # show 3 machines

    sensors = ["temperature", "vibration", "pressure", "rpm", "voltage", "humidity"]
    colors  = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("IoT Sensor Readings Over Time\n(Shaded area = Failure Zone)",
                 fontsize=14, fontweight="bold", y=0.98)

    axes_flat = axes.flatten()

    for idx, (sensor, color) in enumerate(zip(sensors, colors)):
        ax = axes_flat[idx]

        for m_id in machine_ids:
            machine_data = df[df["machine_id"] == m_id].sort_values("cycle")
            ax.plot(machine_data["cycle"], machine_data[sensor],
                    label=f"Machine {m_id}", alpha=0.8, linewidth=1.2)

            # Shade failure zone (last 30 cycles)
            fail_start = machine_data[machine_data["failure"] == 1]["cycle"].min()
            if not pd.isna(fail_start):
                ax.axvspan(fail_start, machine_data["cycle"].max(),
                           alpha=0.15, color="red", label="_nolegend_")

        ax.set_title(sensor.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=9)
        ax.set_ylabel("Normalized Value", fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = "outputs/01_sensor_readings.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sensor readings chart saved: {path}")


def plot_failure_distribution(df: pd.DataFrame):
    """
    Bar chart showing class balance: Normal vs Failure samples.
    Important to understand class imbalance in the dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Failure Class Distribution", fontsize=14, fontweight="bold")

    # ── Left: overall distribution ────────────────────────────────────
    counts = df["failure"].value_counts().sort_index()
    bars = axes[0].bar(["Normal (0)", "Failure (1)"], counts.values,
                       color=["#2ecc71", "#e74c3c"], edgecolor="white",
                       linewidth=1.5, width=0.5)

    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     f"{count:,}\n({count/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    axes[0].set_title("Overall Dataset", fontsize=11)
    axes[0].set_ylabel("Number of Records")
    axes[0].set_ylim(0, counts.max() * 1.2)

    # ── Right: per-machine breakdown ──────────────────────────────────
    per_machine = df.groupby("machine_id")["failure"].sum().reset_index()
    per_machine.columns = ["machine_id", "failure_cycles"]
    bars2 = axes[1].bar(per_machine["machine_id"].astype(str),
                        per_machine["failure_cycles"],
                        color="#e74c3c", alpha=0.8, edgecolor="white")

    axes[1].set_title("Failure Cycles per Machine", fontsize=11)
    axes[1].set_xlabel("Machine ID")
    axes[1].set_ylabel("Number of Failure Cycles")
    axes[1].axhline(per_machine["failure_cycles"].mean(), color="black",
                    linestyle="--", alpha=0.5, label=f"Mean: {per_machine['failure_cycles'].mean():.0f}")
    axes[1].legend()

    plt.tight_layout()
    path = "outputs/02_failure_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Failure distribution chart saved: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Heatmap of the confusion matrix showing:
      - True Positives (predicted failure AND actually failed) ← want this high
      - True Negatives (predicted normal AND actually normal)
      - False Positives (predicted failure but actually normal) ← false alarm
      - False Negatives (predicted normal but actually failed) ← dangerous!
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation: Confusion Matrix", fontsize=14, fontweight="bold")

    # ── Left: raw counts ──────────────────────────────────────────────
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted\nNormal", "Predicted\nFailure"],
                yticklabels=["Actual\nNormal", "Actual\nFailure"],
                ax=axes[0], linewidths=2, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("Raw Counts", fontsize=11)

    # ── Right: normalized (percentages) ──────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Predicted\nNormal", "Predicted\nFailure"],
                yticklabels=["Actual\nNormal", "Actual\nFailure"],
                ax=axes[1], linewidths=2, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("Normalized (Row %)", fontsize=11)

    # Labels
    tn, fp, fn, tp = cm.ravel()
    info = (f"TP={tp} | FP={fp} | FN={fn} | TN={tn}\n"
            f"Recall = {tp/(tp+fn):.3f}  |  Precision = {tp/(tp+fp):.3f}")
    fig.text(0.5, 0.02, info, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = "outputs/03_confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved: {path}")


def plot_feature_importance(model, feature_cols: list, top_n: int = 20):
    """
    Horizontal bar chart of top N most important features.
    Shows WHICH sensors / engineered features matter most for prediction.
    This is key for explainability — critical in industrial AI.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_cols[i] for i in indices]
    top_values   = [importances[i] for i in indices]

    # Color code by type
    def get_color(f):
        if "rolling" in f: return "#3498db"
        if "roc" in f:     return "#e67e22"
        if "ratio" in f:   return "#9b59b6"
        if "_x_" in f:     return "#e74c3c"
        return "#2ecc71"

    colors = [get_color(f) for f in top_features]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_features)), top_values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace("_", " ") for f in top_features[::-1]], fontsize=9)
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Most Important Features\n(Random Forest Gini Importance)",
                 fontsize=13, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Raw sensor"),
        mpatches.Patch(color="#3498db", label="Rolling statistic"),
        mpatches.Patch(color="#e67e22", label="Rate of change"),
        mpatches.Patch(color="#9b59b6", label="Cycle ratio"),
        mpatches.Patch(color="#e74c3c", label="Interaction feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = "outputs/04_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance chart saved: {path}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Side-by-side comparison of actual vs predicted failure labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Actual vs Predicted Failure Labels (Test Set)",
                 fontsize=14, fontweight="bold")

    indices = np.arange(len(y_true))
    sample  = min(300, len(y_true))  # show first 300 samples for clarity

    for ax, labels, title in zip(
        axes,
        [y_true[:sample], y_pred[:sample]],
        ["Actual Labels", "Predicted Labels"]
    ):
        colors = ["#e74c3c" if l == 1 else "#2ecc71" for l in labels]
        ax.bar(range(len(labels)), labels, color=colors,
               width=1.0, edgecolor="none", alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Index (first 300)")
        ax.set_ylabel("Label (0=Normal, 1=Failure)")
        ax.set_ylim(-0.1, 1.3)
        # Add legend patches
        normal_patch  = mpatches.Patch(color="#2ecc71", label="Normal (0)")
        failure_patch = mpatches.Patch(color="#e74c3c", label="Failure (1)")
        ax.legend(handles=[normal_patch, failure_patch])

    plt.tight_layout()
    path = "outputs/05_actual_vs_predicted.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Actual vs predicted chart saved: {path}")


def plot_failure_probability_timeline(model, df_features: pd.DataFrame,
                                       machine_ids: list = None):
    """
    Shows failure probability over time for selected machines.
    The model's confidence that failure is coming rises toward end of life.
    This is the KEY output that maintenance engineers would see on a dashboard.
    """
    from src.feature_engineer import get_feature_columns

    if machine_ids is None:
        machine_ids = df_features["machine_id"].unique()[:4]

    feature_cols = get_feature_columns(df_features)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Failure Probability Timeline per Machine\n"
                 "Red dashed = actual failure zone start",
                 fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()

    for idx, m_id in enumerate(machine_ids[:4]):
        ax = axes_flat[idx]
        machine_data = df_features[df_features["machine_id"] == m_id].sort_values("cycle")

        X_machine = machine_data[feature_cols].values
        proba = model.predict_proba(X_machine)[:, 1]

        cycles = machine_data["cycle"].values
        actual = machine_data["failure"].values

        # Plot failure probability
        ax.plot(cycles, proba, color="#3498db", linewidth=2,
                label="Failure Probability", zorder=3)
        ax.fill_between(cycles, 0, proba, alpha=0.2, color="#3498db")

        # Actual failure zone
        fail_start = machine_data[machine_data["failure"] == 1]["cycle"].min()
        if not pd.isna(fail_start):
            ax.axvline(fail_start, color="#e74c3c", linestyle="--",
                       linewidth=2, label=f"Actual failure zone (cycle {fail_start:.0f})")

        # Alert thresholds
        ax.axhline(0.30, color="#f39c12", linestyle=":", linewidth=1.2,
                   alpha=0.7, label="Warning threshold (30%)")
        ax.axhline(0.70, color="#e74c3c", linestyle=":", linewidth=1.2,
                   alpha=0.7, label="Critical threshold (70%)")

        ax.set_title(f"Machine {m_id}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Failure Probability")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    path = "outputs/06_failure_probability_timeline.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Failure probability timeline saved: {path}")
