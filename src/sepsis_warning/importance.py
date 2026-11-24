"""Grouped permutation importance of the selected model on the validation split.

The model sees 276 columns built from 34 hourly measurements and 4 static values. Shuffling one
column at a time says little, because the eight columns built from one measurement (last value,
rolling mean, hours since seen, ...) can stand in for each other. So the columns are shuffled in
groups, and every column of a group is moved with the same row order:

* ``variable``: all columns built from one measurement (``HR``, ``Lactate``, ...) or one static
  value (``Age``, ``ICULOS``, ...). Answers "which measurements does the model rely on?".
* ``family``: all columns of one kind across every measurement. Answers "does the model rely on
  the measured values, or on whether and when something was measured?". The split is approximate:
  a value column is blank until the first measurement, so it also carries a little of that pattern.

Importance is the drop in hourly average precision when a group is shuffled across all validation
hours, averaged over ``repeats`` shuffles. Average precision needs no threshold and is unchanged by
calibration, so the same number holds for the raw and the calibrated model. A drop near zero means
the model can do without the group, not that the measurement is clinically unimportant; correlated
groups share their importance. Nothing is fitted and no test data is read.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

VALUES = "measured_values"
PATTERN = "measurement_pattern"
STATIC_FAMILY = "static"
# Suffixes from features.build_features that say whether or when a value was measured.
PATTERN_SUFFIXES = ("missing_now", "never_seen", "hours_since_seen", "observations_")
COLUMNS = ["level", "group", "columns", "ap_drop_mean", "ap_drop_std", "baseline_average_precision", "repeats"]


def feature_groups(columns: list[str]) -> dict[str, dict[str, list[str]]]:
    """Column names by measurement (``variable``) and by kind of feature (``family``)."""
    variable: dict[str, list[str]] = {}
    family: dict[str, list[str]] = {}
    for name in columns:
        base, separator, suffix = name.partition("__")
        variable.setdefault(base, []).append(name)
        if not separator:
            kind = STATIC_FAMILY
        elif suffix.startswith(PATTERN_SUFFIXES):
            kind = PATTERN
        else:
            kind = VALUES
        family.setdefault(kind, []).append(name)
    return {"variable": variable, "family": family}


def grouped_permutation_importance(model, table: pd.DataFrame, columns: list[str], *, repeats: int = 3,
                                   seed: int = 42) -> pd.DataFrame:
    """Drop in average precision per group, largest first within each level.

    ``table`` holds the feature columns and ``label``; pass validation rows only.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    labels = table["label"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("permutation importance needs septic and non-septic hours")
    original = table[columns]
    baseline = float(average_precision_score(labels, model.predict_proba(original)[:, 1]))
    rng = np.random.default_rng(seed)
    shuffled = original.copy()
    rows = []
    for level, groups in feature_groups(columns).items():
        for group, names in groups.items():
            values = original[names].to_numpy()
            drops = []
            for _ in range(repeats):
                shuffled[names] = values[rng.permutation(len(values))]
                drops.append(baseline - float(average_precision_score(labels, model.predict_proba(shuffled)[:, 1])))
            shuffled[names] = values
            rows.append({
                "level": level, "group": group, "columns": len(names), "ap_drop_mean": float(np.mean(drops)),
                "ap_drop_std": float(np.std(drops, ddof=1)) if repeats > 1 else 0.0,
                "baseline_average_precision": baseline, "repeats": repeats,
            })
    result = pd.DataFrame(rows, columns=COLUMNS)
    order = {"variable": 0, "family": 1}
    return (result.assign(_level=result["level"].map(order))
            .sort_values(["_level", "ap_drop_mean"], ascending=[True, False], kind="mergesort")
            .drop(columns="_level").reset_index(drop=True))
