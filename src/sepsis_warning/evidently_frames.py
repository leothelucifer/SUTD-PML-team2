"""The columns Evidently compares: the same table from the frozen reference and from the live prediction log.

The scoring service never logs measurement values, only what a monitor may see: the score, whether it crossed the
threshold, whether it raised an alert, how many inputs were out of range, and which inputs were missing in the
latest hour. So the comparison table has exactly those columns, built two ways that must agree:

* ``reference_frame`` from the frozen model scoring hospital A validation in batch (``score_patients``);
* ``prediction_frame`` from the live prediction records.

No patient or stay identifier and no hour is kept, so the table can be stored and shown without exposing a stay.
"""

import pandas as pd

from .data import DYNAMIC

NUMERIC = ["probability", "input_warnings"]
FLAGS = ["above_threshold", "new_alert"]
MISSING = [f"missing_{name}" for name in DYNAMIC]
COLUMNS = NUMERIC + FLAGS + MISSING


def reference_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """From ``score_patients(bundle, patients, DYNAMIC)``: one row per scored stay hour."""
    out = pd.DataFrame({
        "probability": scored["probability"].astype(float),
        "input_warnings": 0.0,   # the batch path has no range warnings; the live path reports its own count
        "above_threshold": scored["hourly_positive"].astype(int),
        "new_alert": scored["alert_event"].astype(int),
    })
    for name in DYNAMIC:
        out[f"missing_{name}"] = scored[name].isna().astype(int).to_numpy()
    return out[COLUMNS].reset_index(drop=True)


def prediction_frame(records: pd.DataFrame) -> pd.DataFrame:
    """From ``read_prediction_log``: one row per scored request (the latest hour of a stay)."""
    if records.empty:
        return pd.DataFrame(columns=COLUMNS)
    missing_now = records["missing_now"].apply(lambda names: set(names) if isinstance(names, (list, tuple, set)) else set())
    out = pd.DataFrame({
        "probability": records["probability"].astype(float),
        "input_warnings": pd.to_numeric(records.get("input_warnings", 0), errors="coerce").fillna(0).astype(float),
        "above_threshold": records["above_threshold"].astype(bool).astype(int),
        "new_alert": records["new_alert"].astype(bool).astype(int),
    })
    for name in DYNAMIC:
        out[f"missing_{name}"] = missing_now.apply(lambda names, n=name: int(n in names)).to_numpy()
    return out[COLUMNS].reset_index(drop=True)
