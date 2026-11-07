"""Causal feature steps for one hourly patient table.

    python scripts/causal_features.py

Runs a six-row example. No patient files are read.
"""

import argparse

import numpy as np
import pandas as pd


WINDOWS = (3, 6, 12)
META = ["patient_id", "hospital", "hour", "ICULOS", "SepsisLabel", "label"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]


def measurement_columns(frame):
    return [name for name in frame.columns if name not in META]


def handle_missing_values(frame):
    """Carry the last observed value forward. Never fill from a later hour."""
    filled = frame.copy()
    for name in measurement_columns(filled):
        filled[name] = filled[name].ffill()
    return filled


def hours_since_last_lab(frame, lab_columns):
    """Hours since any lab was measured, using only rows up to the current hour."""
    lab_columns = list(lab_columns)
    missing = [name for name in lab_columns if name not in frame.columns]
    if not lab_columns or missing:
        raise ValueError(f"lab columns must be present on the frame, missing {missing}")
    hour = np.arange(len(frame), dtype=float)
    observed = frame.loc[:, lab_columns].notna().any(axis=1).to_numpy()
    last_seen = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
    return pd.Series(hour - last_seen, index=frame.index, name="hours_since_last_lab")


def causal_rolling_features(frame, columns, windows=WINDOWS):
    """Last value, rolling mean and standard deviation, and change over each window.

    Hour t uses rows 1..t of this patient only. Means and counts use observed
    values. The last value and the change use a forward fill.
    """
    windows = tuple(windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("windows must be positive")
    columns = list(columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    hour = np.arange(len(frame), dtype=float)
    filled = handle_missing_values(frame.loc[:, columns])
    built = {}
    for name in columns:
        raw = frame[name]
        observed = raw.notna().to_numpy()
        last = filled[name]
        last_seen = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
        built[f"{name}__last"] = last.to_numpy()
        built[f"{name}__missing_now"] = (~observed).astype(float)
        built[f"{name}__never_seen"] = last.isna().to_numpy().astype(float)
        built[f"{name}__hours_since_seen"] = hour - last_seen
        for window in windows:
            roll = raw.rolling(window, min_periods=1)
            built[f"{name}__mean_{window}h"] = roll.mean().to_numpy()
            built[f"{name}__std_{window}h"] = roll.std(ddof=0).to_numpy()
            built[f"{name}__change_{window}h"] = (last - last.shift(window)).to_numpy()
            built[f"{name}__observations_{window}h"] = roll.count().to_numpy()
    return pd.DataFrame(built, index=frame.index)


def build_feature_frame(frame, lab_columns):
    """Shared entry point used by both training and later scoring."""
    features = causal_rolling_features(frame, measurement_columns(frame))
    features["hours_since_last_lab"] = hours_since_last_lab(frame, lab_columns)
    for name in META:
        if name in frame.columns:
            features[name] = frame[name].to_numpy()
    return features


def utility_score(labels, decisions):
    """PhysioNet 2019 utility for one patient."""
    labels = np.asarray(labels)
    decisions = np.asarray(decisions).astype(bool)
    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have the same length")
    if not labels.any():
        positive = np.full(len(labels), -0.05)
        negative = np.zeros(len(labels))
    else:
        # The challenge label is already shifted six hours earlier. Onset here
        # follows the published scorer: first positive row, plus six.
        onset = int(np.argmax(labels)) + 6
        delta = np.arange(len(labels)) - onset
        positive = np.where(
            delta <= -6, np.maximum((delta + 12) / 6, -0.05), (3 - delta) / 9
        )
        negative = np.where(delta <= -6, 0.0, -2 * (delta + 6) / 9)
        positive = np.where(delta <= 3, positive, 0)
        negative = np.where(delta <= 3, negative, 0)
    return float(np.where(decisions, positive, negative).sum())


def leakage_check_corrupt_next_hour(frame):
    """Corrupting hour t+1 must leave the features at hours up to t unchanged."""
    if len(frame) < 2:
        raise ValueError("need at least two hourly rows")
    labs = [name for name in LABS if name in frame.columns]
    if not labs:
        raise ValueError("frame needs at least one lab column")
    t = max(0, len(frame) // 2 - 1)
    before = build_feature_frame(frame, labs)
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[t + 1], measurement_columns(corrupted)] = 9999
    if "SepsisLabel" in corrupted.columns:
        corrupted.loc[corrupted.index[t + 1], "SepsisLabel"] = 1 - int(
            corrupted.iloc[t + 1]["SepsisLabel"]
        )
    after = build_feature_frame(corrupted, labs)
    pd.testing.assert_frame_equal(before.iloc[: t + 1], after.iloc[: t + 1])
    return True


def example_frame():
    return pd.DataFrame({
        "HR": [80, np.nan, np.nan, 90, 90, 100],
        "Lactate": [np.nan, np.nan, 1.5, np.nan, 2.0, np.nan],
        "SepsisLabel": [0, 0, 0, 0, 1, 1],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    frame = example_frame()
    filled = handle_missing_values(frame)
    features = build_feature_frame(frame, ["Lactate"])
    score = utility_score([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1])
    leakage_check_corrupt_next_hour(frame)
    print("Forward-filled HR:", filled["HR"].tolist())
    print("Hours since last lab:", features["hours_since_last_lab"].tolist())
    print("HR mean over 3 hours:", features["HR__mean_3h"].round(2).tolist())
    print(f"Utility score: {score:.4f}")
    print("Next-hour leakage check passed.")


if __name__ == "__main__":
    main()
