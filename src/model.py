"""
model.py
--------
Trains a Random Forest Classifier to predict machine failure.

Why Random Forest?
  - Works well with tabular sensor data out-of-the-box
  - Handles non-linear relationships between sensors and failure
  - Provides feature importance (explains WHY it thinks failure is coming)
  - Does NOT require GPU
  - Fast to train, easy to interpret
  - Industry-standard baseline for predictive maintenance

Model task:
  Binary Classification:
    Input:  sensor readings + engineered features
    Output: 0 = No failure expected
            1 = Failure predicted soon (within 30 cycles)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from src.feature_engineer import get_feature_columns


# ─── MODEL CONFIGURATION ─────────────────────────────────────────────
MODEL_CONFIG = {
    "n_estimators": 150,      # number of decision trees
    "max_depth": 12,          # max depth of each tree (prevents overfitting)
    "min_samples_split": 5,   # min samples to split a node
    "min_samples_leaf": 2,    # min samples in leaf node
    "class_weight": "balanced", # handles class imbalance (fewer failures than normal)
    "random_state": 42,
    "n_jobs": -1              # use all CPU cores
}

TEST_SIZE = 0.20    # 20% data for testing
RANDOM_STATE = 42


def prepare_train_test(df: pd.DataFrame):
    """
    Splits dataset into training and test sets.
    
    IMPORTANT: We split by MACHINE, not by row.
    Why? If we split rows randomly, the model would see data from the same
    machine in both train and test — that's data leakage (cheating).
    
    Instead: Machines 1-8 for training, Machines 9-10 for testing.
    This tests on truly UNSEEN machines — realistic evaluation.
    """
    feature_cols = get_feature_columns(df)
    
    # Get unique machine IDs
    machine_ids = df["machine_id"].unique()
    n_test_machines = max(1, int(len(machine_ids) * TEST_SIZE))
    
    # Sort and split machine IDs
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(machine_ids)
    test_machines = machine_ids[:n_test_machines]
    train_machines = machine_ids[n_test_machines:]
    
    # Split dataset
    train_df = df[df["machine_id"].isin(train_machines)]
    test_df  = df[df["machine_id"].isin(test_machines)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df["failure"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["failure"].values
    
    print(f"\n  Train set: {len(X_train)} samples "
          f"({len(train_machines)} machines)")
    print(f"  Test set:  {len(X_test)} samples "
          f"({len(test_machines)} machines)")
    print(f"  Features:  {len(feature_cols)}")
    print(f"\n  Train failure distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = "Failure" if u == 1 else "Normal "
        print(f"    {label} (class {u}): {c} samples ({c/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(df: pd.DataFrame):
    """
    Trains the Random Forest model on the prepared dataset.

    Parameters:
        df: Feature-engineered DataFrame

    Returns:
        model:        Trained Random Forest model
        X_test:       Test feature matrix (for evaluation)
        y_test:       True labels for test set
        feature_cols: List of feature names
    """
    print("\n  Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)

    print("\n  Training Random Forest Classifier...")
    print(f"  Config: {MODEL_CONFIG}")

    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)

    print("  ✓ Model training complete!")

    # Quick cross-validation on training set
    print("\n  Running 5-fold cross-validation on training data...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"  CV F1 Scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean CV F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, X_test, y_test, feature_cols


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the trained model on the test set.

    Returns a dict with accuracy, precision, recall, and F1-score.
    
    Metric guide:
      Accuracy:  % of total predictions correct (can be misleading with imbalance)
      Precision: Of all predicted failures, how many were real? (avoid false alarms)
      Recall:    Of all real failures, how many did we catch? (most important!)
      F1-Score:  Harmonic mean of precision and recall (balanced metric)
    
    For predictive maintenance: HIGH RECALL is more important than precision.
    Missing a real failure (false negative) is worse than a false alarm.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "y_pred":    y_pred,
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    print("\n  ─── MODEL EVALUATION RESULTS ────────────────────")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}  ← (most important for safety)")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))
    print("  ─────────────────────────────────────────────────")

    return metrics


def save_model(model, path: str = "models/predictive_model.pkl"):
    """Saves trained model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {path}")


def load_model(path: str = "models/predictive_model.pkl"):
    """Loads a previously trained model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  ✓ Model loaded from {path}")
    return model


# ─── QUICK TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from src.data_loader import load_data
    from src.preprocessor import preprocess_data
    from src.feature_engineer import engineer_features

    df_raw = load_data(save_csv=False)
    df_clean = preprocess_data(df_raw)
    df_features = engineer_features(df_clean)
    model, X_test, y_test, feature_cols = train_model(df_features)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
