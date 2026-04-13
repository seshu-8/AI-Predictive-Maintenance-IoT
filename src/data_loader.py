"""
data_loader.py
--------------
Generates simulated IoT sensor data that mimics real industrial machine readings.
Simulates: Temperature, Vibration, Pressure, RPM, Voltage, and Humidity sensors.

In a real project, this file would load data from:
  - CSV files (e.g., NASA CMAPSS dataset from Kaggle)
  - SQL database
  - AWS IoT Core / Azure IoT Hub
  - Kafka stream

For student use: We simulate realistic data so you can run this WITHOUT hardware.
"""

import numpy as np
import pandas as pd
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────
SEED = 42
N_MACHINES = 10          # number of virtual IoT machines
CYCLES_PER_MACHINE = 300 # number of time cycles per machine
FAILURE_WINDOW = 30      # machine "fails" in the last 30 cycles of its life

# ─── SENSOR NAMES ────────────────────────────────────────────────────
SENSORS = [
    "temperature",   # °C
    "vibration",     # mm/s
    "pressure",      # bar
    "rpm",           # rotations per minute
    "voltage",       # V
    "humidity"       # %
]

def _generate_machine_data(machine_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generates sensor readings for one machine over CYCLES_PER_MACHINE cycles.
    As the machine approaches failure, sensor values degrade gradually.
    """
    n = CYCLES_PER_MACHINE

    # Time index
    cycles = np.arange(1, n + 1)

    # ── Degradation factor: starts at 0, increases toward 1 (failure) ──
    # This simulates how machines degrade over time
    degradation = (cycles / n) ** 2  # non-linear degradation curve

    # ── Simulate each sensor ──────────────────────────────────────────
    # Normal operating range + noise + degradation effect
    temperature = (
        75 + 20 * degradation               # rises from 75°C to 95°C
        + rng.normal(0, 2, n)               # random noise
    )

    vibration = (
        2 + 8 * degradation                 # rises from 2 to 10 mm/s
        + rng.normal(0, 0.5, n)
    )

    pressure = (
        100 - 15 * degradation              # drops from 100 to 85 bar
        + rng.normal(0, 1.5, n)
    )

    rpm = (
        3000 - 500 * degradation            # drops from 3000 to 2500
        + rng.normal(0, 50, n)
    )

    voltage = (
        220 - 10 * degradation              # drops slightly
        + rng.normal(0, 2, n)
    )

    humidity = (
        50 + 20 * degradation               # rises as seals degrade
        + rng.normal(0, 3, n)
    )

    # ── Failure label: 1 if within last FAILURE_WINDOW cycles ─────────
    # This is what the model will LEARN to predict
    failure_label = np.where(cycles >= (n - FAILURE_WINDOW), 1, 0)

    # ── Assemble into DataFrame ───────────────────────────────────────
    df = pd.DataFrame({
        "machine_id": machine_id,
        "cycle": cycles,
        "temperature": np.round(temperature, 2),
        "vibration": np.round(vibration, 3),
        "pressure": np.round(pressure, 2),
        "rpm": np.round(rpm, 1),
        "voltage": np.round(voltage, 2),
        "humidity": np.round(humidity, 2),
        "failure": failure_label
    })

    return df


def load_data(save_csv: bool = True) -> pd.DataFrame:
    """
    Main function: generates data for all machines and combines them.

    Parameters:
        save_csv (bool): If True, saves raw data to data/raw/sensor_data.csv

    Returns:
        pd.DataFrame: Full dataset with all machines and sensor readings
    """
    rng = np.random.default_rng(SEED)

    print(f"  Generating data for {N_MACHINES} virtual machines...")
    print(f"  Each machine has {CYCLES_PER_MACHINE} time cycles.")
    print(f"  Failure label = 1 in the last {FAILURE_WINDOW} cycles.")

    all_machines = []
    for machine_id in range(1, N_MACHINES + 1):
        machine_df = _generate_machine_data(machine_id, rng)
        all_machines.append(machine_df)
        print(f"    ✓ Machine {machine_id:02d} data generated")

    # Combine all machine data
    df = pd.concat(all_machines, ignore_index=True)

    # Save to CSV for reference
    if save_csv:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/sensor_data.csv", index=False)
        print(f"\n  ✓ Raw data saved to: data/raw/sensor_data.csv")

    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Failure class distribution:")
    print(df["failure"].value_counts().to_string())

    return df


# ─── QUICK TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
