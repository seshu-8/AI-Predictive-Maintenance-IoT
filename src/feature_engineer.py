"""
feature_engineer.py
-------------------
Creates meaningful features from raw sensor readings.

Why feature engineering matters:
  - Raw sensor readings alone are often not enough for good predictions
  - Rolling averages smooth out noise and reveal trends
  - Rate of change (derivative) captures HOW FAST a sensor is changing
  - These engineered features often matter MORE than raw readings

Features created per sensor:
  1. Rolling Mean (window=10)    → smoothed trend
  2. Rolling Std  (window=10)    → variability / instability
  3. Rolling Min  (window=10)    → worst-case in recent window
  4. Rolling Max  (window=10)    → peak values
  5. Rate of Change              → first derivative (is it getting worse?)

Additional features:
  6. Cycle ratio                 → how far is machine into its life? (0 to 1)
  7. Temperature × Vibration     → interaction feature (compound stress)
  8. Pressure × RPM              → mechanical load proxy
"""

import pandas as pd
import numpy as np
import os

SENSORS = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "voltage",
    "humidity"
]

ROLLING_WINDOW = 10  # use last 10 cycles for rolling stats


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling window statistics for each sensor.
    Computed per machine (grouped) to avoid cross-machine contamination.
    """
    print("  Adding rolling window features (window=10 cycles)...")

    for sensor in SENSORS:
        grouped = df.groupby("machine_id")[sensor]

        # Rolling mean: smoothed signal
        df[f"{sensor}_rolling_mean"] = grouped.transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
        )

        # Rolling std: instability indicator
        df[f"{sensor}_rolling_std"] = grouped.transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
        )

        # Rolling min: lowest value in window
        df[f"{sensor}_rolling_min"] = grouped.transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).min()
        )

        # Rolling max: highest value in window
        df[f"{sensor}_rolling_max"] = grouped.transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).max()
        )

        print(f"    ✓ {sensor}: mean, std, min, max added")

    return df


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds first-order difference (rate of change) for each sensor.
    Tells the model: is this sensor VALUE increasing or decreasing?
    A rapidly rising temperature is more alarming than a stable high one.
    """
    print("  Adding rate-of-change (derivative) features...")

    for sensor in SENSORS:
        df[f"{sensor}_roc"] = df.groupby("machine_id")[sensor].diff().fillna(0)
        print(f"    ✓ {sensor}_roc added")

    return df


def add_cycle_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cycle_ratio: how far through its lifecycle is each machine?
    
    cycle_ratio = current_cycle / max_cycle_for_this_machine
    
    Value of 0 = machine is new
    Value of 1 = machine at end of life
    
    This is a powerful feature that tells the model machine "age".
    """
    print("  Adding cycle ratio (machine age indicator)...")

    max_cycles = df.groupby("machine_id")["cycle"].transform("max")
    df["cycle_ratio"] = df["cycle"] / max_cycles
    print("    ✓ cycle_ratio added")

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features between correlated sensors.
    These capture compound stress effects that individual sensors miss.
    
    Examples:
      - High temperature + high vibration = serious mechanical stress
      - Low pressure + high RPM = cavitation risk
    """
    print("  Adding sensor interaction features...")

    # Thermal + mechanical stress
    df["temp_x_vibration"] = df["temperature"] * df["vibration"]

    # Mechanical load proxy
    df["pressure_x_rpm"] = df["pressure"] * df["rpm"]

    # Power quality index
    df["voltage_x_rpm"] = df["voltage"] * df["rpm"]

    print("    ✓ temp_x_vibration, pressure_x_rpm, voltage_x_rpm added")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Parameters:
        df: Preprocessed DataFrame from preprocessor.py

    Returns:
        df_features: DataFrame with all engineered features
    """
    print("\n  Running feature engineering pipeline...")
    df = df.copy()

    # Step 1: Rolling statistics
    df = add_rolling_features(df)

    # Step 2: Rate of change
    df = add_rate_of_change(df)

    # Step 3: Cycle ratio
    df = add_cycle_ratio(df)

    # Step 4: Interaction features
    df = add_interaction_features(df)

    # Drop rows where rolling features might be NaN (first few cycles)
    df = df.dropna().reset_index(drop=True)

    # Save engineered features
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/engineered_features.csv", index=False)

    # Count all features
    feature_cols = [c for c in df.columns if c not in ["machine_id", "cycle", "failure"]]
    print(f"\n  ✓ Total features created: {len(feature_cols)}")
    print(f"  ✓ Final dataset shape: {df.shape}")
    print(f"  ✓ Saved to data/processed/engineered_features.csv")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns the list of feature column names (excludes metadata and label)."""
    return [c for c in df.columns if c not in ["machine_id", "cycle", "failure"]]


# ─── QUICK TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from src.data_loader import load_data
    from src.preprocessor import preprocess_data

    df_raw = load_data(save_csv=False)
    df_clean = preprocess_data(df_raw)
    df_features = engineer_features(df_clean)

    print("\nFeature columns:")
    feat_cols = get_feature_columns(df_features)
    for col in feat_cols:
        print(f"  - {col}")
