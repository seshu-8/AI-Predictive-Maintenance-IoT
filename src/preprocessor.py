"""
preprocessor.py
---------------
Cleans and prepares raw sensor data for machine learning.

Steps done here:
  1. Handle missing values
  2. Remove outliers (IQR method)
  3. Clip sensor values to realistic ranges
  4. Normalize/scale features (MinMaxScaler)
  5. Save processed data

Why preprocessing matters:
  - Raw sensor data has noise, errors, and outliers
  - ML models perform badly on dirty data
  - Normalization ensures no single sensor dominates the model
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler


# ─── SENSOR VALID RANGES ─────────────────────────────────────────────
# These represent physically realistic boundaries for each sensor
SENSOR_RANGES = {
    "temperature": (40, 120),    # °C
    "vibration":   (0, 20),      # mm/s
    "pressure":    (50, 150),    # bar
    "rpm":         (1000, 4000), # rpm
    "voltage":     (180, 260),   # V
    "humidity":    (10, 100),    # %
}

SENSOR_COLUMNS = list(SENSOR_RANGES.keys())


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sensor readings using forward-fill (last known reading).
    In real IoT systems, sensors sometimes drop packets — this handles that.
    """
    initial_missing = df[SENSOR_COLUMNS].isnull().sum().sum()

    if initial_missing > 0:
        print(f"  Found {initial_missing} missing values. Applying forward-fill...")
        # Forward fill: use previous valid reading
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].fillna(method="ffill")
        # Backward fill for any remaining (start of series)
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].fillna(method="bfill")
        # If still missing, fill with column median
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].fillna(df[SENSOR_COLUMNS].median())
        print(f"  ✓ Missing values handled.")
    else:
        print(f"  ✓ No missing values found.")

    return df


def clip_to_valid_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips sensor readings to physically valid ranges.
    A temperature of -500°C is impossible — this removes such errors.
    """
    print("  Clipping sensors to valid physical ranges...")
    for sensor, (low, high) in SENSOR_RANGES.items():
        before_out = ((df[sensor] < low) | (df[sensor] > high)).sum()
        df[sensor] = df[sensor].clip(low, high)
        if before_out > 0:
            print(f"    {sensor}: {before_out} out-of-range values clipped")

    print("  ✓ Range clipping complete.")
    return df


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Removes extreme outliers using IQR (Interquartile Range) method.
    Uses multiplier=3.0 (lenient) to preserve valid degradation patterns.

    IQR Method:
      Q1 = 25th percentile
      Q3 = 75th percentile
      IQR = Q3 - Q1
      Outlier = value < Q1 - 3*IQR  OR  value > Q3 + 3*IQR
    """
    print(f"  Removing outliers using IQR (multiplier={multiplier})...")
    initial_rows = len(df)

    # Calculate IQR per machine (so each machine's degradation is treated fairly)
    mask = pd.Series([True] * len(df), index=df.index)

    for sensor in SENSOR_COLUMNS:
        Q1 = df[sensor].quantile(0.25)
        Q3 = df[sensor].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        sensor_mask = (df[sensor] >= lower) & (df[sensor] <= upper)
        mask = mask & sensor_mask

    df = df[mask].copy()
    removed = initial_rows - len(df)
    print(f"  ✓ Removed {removed} outlier rows ({removed/initial_rows*100:.2f}%)")
    return df


def scale_features(df: pd.DataFrame, save_scaler: bool = True) -> pd.DataFrame:
    """
    Applies MinMaxScaler to normalize sensor values to [0, 1] range.
    
    Why MinMax (not StandardScaler)?
    - Sensor data has known physical bounds
    - We want to preserve the shape of degradation curves
    - Easier to interpret: 0 = min reading, 1 = max reading
    """
    print("  Scaling sensor features to [0, 1] range...")

    scaler = MinMaxScaler()
    df[SENSOR_COLUMNS] = scaler.fit_transform(df[SENSOR_COLUMNS])

    # Save scaler so we can use it later for real predictions
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("  ✓ Scaler saved to models/scaler.pkl")

    print("  ✓ Feature scaling complete.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    Runs all steps in correct order.

    Parameters:
        df: Raw sensor DataFrame from data_loader.py

    Returns:
        df_clean: Cleaned and normalized DataFrame
    """
    print("\n  Running preprocessing pipeline...")
    df = df.copy()

    # Step 1: Handle missing values
    df = handle_missing_values(df)

    # Step 2: Clip to valid sensor ranges
    df = clip_to_valid_range(df)

    # Step 3: Remove extreme outliers
    df = remove_outliers_iqr(df)

    # Step 4: Normalize sensor values
    df = scale_features(df)

    # Step 5: Sort by machine and cycle for time-series integrity
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_data.csv", index=False)
    print("  ✓ Processed data saved to data/processed/clean_data.csv")
    print(f"  Final shape: {df.shape}")

    return df


# ─── QUICK TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    df_raw = load_data(save_csv=False)
    df_clean = preprocess_data(df_raw)
    print("\nCleaned data sample:")
    print(df_clean.head())
    print("\nValue ranges after scaling:")
    print(df_clean[SENSOR_COLUMNS].describe())
