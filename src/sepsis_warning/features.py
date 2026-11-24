"""Patient-local features: feature[t] depends only on observations up to t.

Training, replay and the scoring service call ``build_features`` with one
trailing window (default 6 hours). ``features_for_windows`` and the training-
median fallback below are experimental helpers: they are tested and used by the
demo script and by signature features, but ``train`` does not call them.
``train`` still fills leftover blanks with the model imputer.
"""

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, Patient


def ordered_stay(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort one stay by ICULOS and require consecutive hourly rows."""
    if "ICULOS" not in frame.columns or frame["ICULOS"].isna().any():
        raise ValueError("ICULOS is required for every row")
    ordered = frame.sort_values("ICULOS", kind="mergesort").reset_index(drop=True)
    hours = ordered["ICULOS"].to_numpy(dtype=float)
    if len(hours) > 1 and not np.allclose(np.diff(hours), 1):
        raise ValueError("ICULOS must be consecutive hourly after sorting")
    return ordered


def build_features(frame: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Features at hour t use this stay only, through hour t.

    The stay is sorted by ICULOS. Hours must be consecutive after that sort.
    The last observed value is carried forward until the stay ends or a new
    measurement arrives. There is no expiry. A missing value is never filled
    from a later hour.

    ``hours_since_seen`` is an ICU-hour difference from ICULOS. The trailing
    mean, std, observation count and change use the last ``window`` rows after
    sorting, not a wall-clock ICU-hour window. ``change`` is blank for the first
    ``window`` rows of every stay.
    """
    if window < 1:
        raise ValueError("window must be positive")
    ordered = ordered_stay(frame)
    columns = {}
    hour = ordered["ICULOS"].to_numpy(dtype=float)
    for name in DYNAMIC:
        raw = ordered[name]
        observed = raw.notna().to_numpy()
        # No backfill or interpolation across future measurements.
        last = raw.ffill()
        last_seen_hour = pd.Series(np.where(observed, hour, np.nan)).ffill().to_numpy()
        roll = raw.rolling(window, min_periods=1)
        columns[f"{name}__last"] = last
        columns[f"{name}__missing_now"] = (~observed).astype(float)
        columns[f"{name}__never_seen"] = last.isna().astype(float)
        columns[f"{name}__hours_since_seen"] = hour - last_seen_hour
        columns[f"{name}__mean_{window}h"] = roll.mean()
        columns[f"{name}__std_{window}h"] = roll.std(ddof=0)
        columns[f"{name}__change_{window}h"] = last - last.shift(window)
        columns[f"{name}__observations_{window}h"] = roll.count()
    for name in STATIC:
        columns[name] = ordered[name].ffill()
    return pd.DataFrame(columns).astype(np.float32)


def feature_table(patients: list[Patient], window: int) -> pd.DataFrame:
    chunks = []
    for patient in patients:
        ordered = ordered_stay(patient.frame)
        features = build_features(ordered, window)
        features["patient_id"] = patient.patient_id
        features["hospital"] = patient.hospital
        features["hour"] = ordered["ICULOS"].to_numpy()
        # The challenge has ALREADY shifted its labels six hours earlier.
        features["label"] = ordered["SepsisLabel"].to_numpy(dtype=int)
        chunks.append(features)
    return pd.concat(chunks, ignore_index=True)


META = ["patient_id", "hospital", "hour", "label"]


def select_saved_features(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Keep the saved column list, in that order. Reject a stay that cannot supply it."""
    if not feature_columns:
        raise ValueError("saved model has no feature columns")
    missing = [name for name in feature_columns if name not in features.columns]
    if missing:
        shown = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(f"saved feature columns are missing from this stay: {shown}{extra}")
    return features.loc[:, feature_columns]


def _window_column(name: str) -> bool:
    return any(token in name for token in ("__mean_", "__std_", "__change_", "__observations_"))


def features_for_windows(frame: pd.DataFrame, windows: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    """Experimental multi-window builder. ``train`` and serve use one window only."""
    if not windows:
        raise ValueError("windows must be positive")
    built = [build_features(frame, window) for window in windows]
    base = [name for name in built[0].columns if not _window_column(name)]
    pieces = [built[0].loc[:, base]]
    for window, features in zip(windows, built):
        names = [name for name in features.columns if name.endswith(f"_{window}h")]
        pieces.append(features.loc[:, names])
    return pd.concat(pieces, axis=1)


def fit_training_fallback(frames: list[pd.DataFrame]) -> pd.Series:
    """Experimental: median of observed measurements. Pass training tables only.

    ``train`` does not call this. It fills leftover blanks with the model
    imputer after ``build_features``. A column that is blank in every training
    row stays blank. Rows that are not included here cannot change the medians.
    Carry-forward still has no expiry; this function only supplies a value
    before the first observation.
    """
    if not frames:
        raise ValueError("training fallback needs at least one table")
    columns = [name for name in DYNAMIC if any(name in frame.columns for frame in frames)]
    if not columns:
        raise ValueError("training tables have no measurement columns")
    observed = pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)
    return observed.median()


def apply_training_fallback(frame: pd.DataFrame, fallback: pd.Series) -> pd.DataFrame:
    """Experimental: forward-fill inside each stay, then fill what is still missing.

    Does not refit the medians and does not modify ``frame``. A later hour never
    fills an earlier blank. Stays are split by hospital and patient_id when both
    are present, otherwise by patient_id alone. Row order is preserved.
    """
    columns = [name for name in DYNAMIC if name in frame.columns]
    filled = frame.copy()
    if not columns:
        return filled

    def fill_columns(stay: pd.DataFrame) -> pd.DataFrame:
        out = stay.copy()
        for name in columns:
            out[name] = out[name].ffill()
            if name in fallback.index:
                out[name] = out[name].fillna(fallback[name])
        return out

    if {"hospital", "patient_id"} <= set(filled.columns):
        keys = ["hospital", "patient_id"]
    elif "patient_id" in filled.columns:
        keys = ["patient_id"]
    else:
        return fill_columns(filled)

    if filled.groupby(keys, sort=False).ngroups <= 1:
        return fill_columns(filled)

    order = np.arange(len(filled))
    filled["_row_order"] = order
    pieces = [fill_columns(stay) for _, stay in filled.groupby(keys, sort=False)]
    return pd.concat(pieces).sort_values("_row_order").drop(columns="_row_order")
