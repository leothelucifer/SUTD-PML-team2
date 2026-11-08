"""Causal feature steps for one hourly patient table.

    python scripts/causal_features.py

Runs a small example. No patient files are read. This script does not replace
the package used by training and replay.
"""

import argparse

import numpy as np
import pandas as pd


WINDOWS = (3, 6, 12)
META = ["patient_id", "hospital", "hour", "ICULOS", "SepsisLabel", "label"]
EXCLUDED = META + ["Unit1", "Unit2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
FEATURE_DTYPE = np.float32


def measurement_columns(frame):
    return [name for name in frame.columns if name not in EXCLUDED]


def handle_missing_values(frame):
    """Carry the last observed value forward inside this table. Never fill from a later hour."""
    filled = frame.copy()
    for name in measurement_columns(filled):
        filled[name] = filled[name].ffill()
    return filled


def fit_fallback(frames):
    """Median of observed measurements. Pass training tables only.

    A column that is blank in every training row stays blank. Rows in the
    tables passed here are the only rows that affect the medians.
    """
    columns = measurement_columns(frames[0])
    observed = pd.concat([frame.loc[:, columns] for frame in frames], ignore_index=True)
    return observed.median(numeric_only=True)


def apply_fallback(frame, fallback):
    """Fill values still missing after forward fill. Does not refit the medians."""
    filled = handle_missing_values(frame)
    for name, value in fallback.items():
        if name in filled.columns:
            filled[name] = filled[name].fillna(value)
    return filled


def _hour_index(frame):
    return np.arange(len(frame), dtype=float)


def per_lab_recency(frame, lab_columns):
    """Per lab: hours since the last observation, missing now, and never seen."""
    lab_columns = list(lab_columns)
    missing = [name for name in lab_columns if name not in frame.columns]
    if not lab_columns or missing:
        raise ValueError(f"lab columns must be present on the frame, missing {missing}")
    hour = _hour_index(frame)
    built = {}
    for name in lab_columns:
        observed = frame[name].notna().to_numpy()
        last_seen = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
        built[f"{name}__hours_since_seen"] = hour - last_seen
        built[f"{name}__missing_now"] = (~observed).astype(float)
        built[f"{name}__never_seen"] = pd.Series(np.where(observed, hour, np.nan)).ffill().isna().to_numpy().astype(float)
    return pd.DataFrame(built, index=frame.index)


def hours_since_last_lab(frame, lab_columns):
    """Hours since any listed lab was measured, using only rows up to the current hour."""
    lab_columns = list(lab_columns)
    missing = [name for name in lab_columns if name not in frame.columns]
    if not lab_columns or missing:
        raise ValueError(f"lab columns must be present on the frame, missing {missing}")
    hour = _hour_index(frame)
    observed = frame.loc[:, lab_columns].notna().any(axis=1).to_numpy()
    last_seen = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
    return pd.Series(hour - last_seen, index=frame.index, name="hours_since_last_lab")


def feature_names(columns, lab_columns, windows=WINDOWS):
    """Stable column order for one shared feature table."""
    names = []
    for name in columns:
        names.extend([
            f"{name}__last",
            f"{name}__missing_now",
            f"{name}__never_seen",
            f"{name}__hours_since_seen",
        ])
        for window in windows:
            names.extend([
                f"{name}__mean_{window}h",
                f"{name}__std_{window}h",
                f"{name}__change_{window}h",
                f"{name}__observations_{window}h",
            ])
    if lab_columns:
        names.append("hours_since_last_lab")
    return names


def feature_contract(columns, lab_columns, windows=WINDOWS):
    return {
        "windows": list(windows),
        "columns": feature_names(columns, lab_columns, windows),
        "dtype": "float32",
    }


def causal_rolling_features(frame, columns, windows=WINDOWS):
    """Last value, rolling mean, standard deviation, count, and change over each window.

    Hour t uses rows 1..t of this patient only. Means and counts use observed
    values. The last value and the change use a forward fill. A window shorter
    than the history uses the rows that exist.
    """
    windows = tuple(windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("windows must be positive")
    columns = list(columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    hour = _hour_index(frame)
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
    ordered = feature_names(columns, [], windows)
    return pd.DataFrame(built, index=frame.index)[ordered]


def _build_one_patient(frame, lab_columns, windows):
    columns = measurement_columns(frame)
    features = causal_rolling_features(frame, columns, windows)
    present_labs = [name for name in lab_columns if name in frame.columns]
    if present_labs:
        recency = per_lab_recency(frame, present_labs).astype(FEATURE_DTYPE)
        features[recency.columns] = recency
    if lab_columns:
        features["hours_since_last_lab"] = hours_since_last_lab(frame, lab_columns)
    ordered = feature_names(columns, lab_columns, windows)
    features = features[ordered].astype(FEATURE_DTYPE)
    for name in EXCLUDED:
        if name in frame.columns:
            features[name] = frame[name].to_numpy()
    return features


def build_feature_frame(frame, lab_columns, windows=WINDOWS):
    """One entry point. Each patient is built separately, so rows cannot cross patients."""
    lab_columns = [name for name in lab_columns if name in frame.columns]
    if "patient_id" in frame.columns and frame["patient_id"].nunique() > 1:
        parts = [
            _build_one_patient(part, lab_columns, windows)
            for _, part in frame.groupby("patient_id", sort=False)
        ]
        return pd.concat(parts)
    return _build_one_patient(frame, lab_columns, windows)


def utility_components(labels):
    """Hourly PhysioNet 2019 weights. Decisions are hourly, not cooldown alerts."""
    labels = np.asarray(labels)
    if not np.asarray(labels).any():
        return np.full(len(labels), -0.05), np.zeros(len(labels))
    onset = int(np.argmax(labels)) + 6
    delta = np.arange(len(labels)) - onset
    positive = np.where(
        delta <= -6, np.maximum((delta + 12) / 6, -0.05), (3 - delta) / 9
    )
    negative = np.where(delta <= -6, 0.0, -2 * (delta + 6) / 9)
    return np.where(delta <= 3, positive, 0.0), np.where(delta <= 3, negative, 0.0)


def utility_score(labels, decisions):
    """PhysioNet 2019 utility for one patient."""
    labels = np.asarray(labels)
    decisions = np.asarray(decisions).astype(bool)
    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have the same length")
    positive, negative = utility_components(labels)
    return float(np.where(decisions, positive, negative).sum())


def leakage_check_corrupt_next_hour(frame):
    """Corrupting hour t+1 onward must leave features through hour t unchanged.

    The corrupted rows include measurements, the label, and excluded metadata.
    """
    if len(frame) < 2:
        raise ValueError("need at least two hourly rows")
    labs = [name for name in LABS if name in frame.columns]
    if not labs:
        raise ValueError("frame needs at least one lab column")
    t = max(0, len(frame) // 2 - 1)
    before = build_feature_frame(frame, labs)
    corrupted = frame.copy()
    rows = corrupted.index[t + 1 :]
    corrupted.loc[rows, measurement_columns(corrupted)] = 9999
    if "SepsisLabel" in corrupted.columns:
        corrupted.loc[rows, "SepsisLabel"] = 1 - corrupted.loc[rows, "SepsisLabel"].astype(int)
    for name in ("patient_id", "hospital", "Unit1", "Unit2"):
        if name in corrupted.columns:
            corrupted.loc[rows, name] = 999 if pd.api.types.is_numeric_dtype(corrupted[name]) else "CORRUPTED"
    after = build_feature_frame(corrupted, labs)
    pd.testing.assert_frame_equal(before.iloc[: t + 1], after.iloc[: t + 1])
    return True


def record_isolation_check(left, right):
    """Features for one patient stay the same when another patient is in the table."""
    labs = [name for name in LABS if name in left.columns and name in right.columns]
    alone = build_feature_frame(left, labs)
    together = pd.concat([
        left.assign(patient_id="left"),
        right.assign(patient_id="right"),
    ], ignore_index=True)
    combined = build_feature_frame(together, labs)
    built = combined.loc[combined["patient_id"] == "left", alone.columns]
    pd.testing.assert_frame_equal(alone.reset_index(drop=True), built.reset_index(drop=True))
    return True


def example_frame():
    return pd.DataFrame({
        "patient_id": ["a"] * 6,
        "Unit1": [1, 1, 1, 0, 0, 0],
        "HR": [80, np.nan, np.nan, 90, 90, 100],
        "Lactate": [np.nan, np.nan, 1.5, np.nan, 2.0, np.nan],
        "SepsisLabel": [0, 0, 0, 0, 1, 1],
    })


def check_utility_examples():
    published = utility_score([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1])
    if abs(published - 3.388888888888889) > 1e-9:
        raise AssertionError(f"published utility example returned {published}")
    blank = np.zeros(4, dtype=int)
    if utility_score(blank, np.zeros(4, dtype=int)) != 0:
        raise AssertionError("an all-negative record should score 0 when no alert fires")
    if abs(utility_score(blank, np.ones(4, dtype=int)) - (-0.2)) > 1e-9:
        raise AssertionError("false alerts on a non-septic record should score -0.05 each hour")
    long_label = np.r_[np.zeros(18), np.ones(22)]
    positive, negative = utility_components(long_label)
    if not (positive[28:] == 0).all() or not (negative[28:] == 0).all():
        raise AssertionError("utility should stop three hours after the inferred onset")
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    frame = example_frame()
    filled = handle_missing_values(frame)
    features = build_feature_frame(frame, ["Lactate"])
    again = build_feature_frame(frame, ["Lactate"])
    pd.testing.assert_frame_equal(features, again)
    train = frame.loc[:, ["HR", "Lactate"]].copy()
    train["NeverDrawn"] = np.nan
    fallback = fit_fallback([train])
    held_out = pd.DataFrame({"HR": [np.nan, np.nan], "Lactate": [np.nan, 4.0], "NeverDrawn": [np.nan, np.nan]})
    applied = apply_fallback(held_out, fallback)
    score = check_utility_examples()
    leakage_check_corrupt_next_hour(frame)
    other = frame.copy()
    other["HR"] = 50
    record_isolation_check(frame.drop(columns="patient_id"), other.drop(columns="patient_id"))
    contract = feature_contract(measurement_columns(frame), ["Lactate"])
    print("Forward-filled HR:", filled["HR"].tolist())
    print("Hours since last lactate:", features["Lactate__hours_since_seen"].tolist())
    print("Lactate missing now:", features["Lactate__missing_now"].tolist())
    print("Hours since any lab:", features["hours_since_last_lab"].tolist())
    print("HR mean over 3 hours:", [round(float(value), 2) for value in features["HR__mean_3h"]])
    print("Training median HR:", float(fallback["HR"]))
    print("Held-out HR after fallback:", applied["HR"].tolist())
    print("Unobserved training column stays blank:", bool(applied["NeverDrawn"].isna().all()))
    print(f"Feature columns: {len(contract['columns'])} at {contract['dtype']}")
    print(f"Utility score: {score:.4f}")
    print("Next-hour leakage check passed.")
    print("Record isolation check passed.")


if __name__ == "__main__":
    main()
