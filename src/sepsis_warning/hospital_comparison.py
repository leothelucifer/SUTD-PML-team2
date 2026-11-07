"""Descriptive comparison of hospital A and hospital B (Person 1, Task 2).

Everything here is DESCRIPTIVE. Nothing computed from hospital B is used to
choose features, models or thresholds; B stays an untouched external test set.
All outputs are aggregates (no patient-level rows or trajectories).

Recreate every figure and table with one command:

    python -m sepsis_warning compare-hospitals \
        --data data/physionet/combined.parquet --output reports/figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import SIRS_ALERT_COUNT, SIRS_COLUMNS, sirs_criteria_from_last_seen
from .data import DYNAMIC

KEY = ["hospital", "patient_id"]
HOSPITALS = ["A", "B"]
COLORS = {"A": "#2a6fdb", "B": "#e07b39"}
# Legend names; run_comparison switches A to "Hospital A (train)" when it
# restricts A to the training split.
LABELS = {"A": "Hospital A", "B": "Hospital B"}
AGE_BINS = [0, 45, 65, 80, np.inf]
AGE_LABELS = ["<45", "45-64", "65-79", "80+"]
# Plot ranges cut off rare extreme values so the shape is visible.
# Values outside are NOT removed from the data; they are only off-chart.
VITAL_RANGES = {"HR": (20, 200), "MAP": (20, 160), "Temp": (33, 42), "Resp": (0, 50)}
# Bin widths match how values are recorded (whole numbers; Temp often
# converted from Fahrenheit), so the histograms show no comb artefacts.
VITAL_BIN_WIDTH = {"HR": 2, "MAP": 2, "Temp": 0.25, "Resp": 1}
VITAL_UNITS = {"HR": "beats/min", "MAP": "mmHg", "Temp": "°C", "Resp": "breaths/min"}
RECREATE = (
    "python -m sepsis_warning compare-hospitals --data data/physionet/combined.parquet "
    "--splits data/physionet/splits.csv --output reports/figures"
)


def load_hours(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = {"hospital", "patient_id", "ICULOS", "SepsisLabel"} - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame.sort_values(KEY + ["ICULOS"], kind="mergesort").reset_index(drop=True)


def add_sirs(hours: pd.DataFrame) -> pd.DataFrame:
    """Add sirs_count / sirs_alert using a forward fill INSIDE each stay only."""
    last_seen = hours.groupby(KEY, sort=False)[SIRS_COLUMNS].ffill()
    count = sirs_criteria_from_last_seen(last_seen).sum(axis=1)
    return hours.assign(sirs_count=count.to_numpy(), sirs_alert=(count >= SIRS_ALERT_COUNT).to_numpy())


def stay_table(hours: pd.DataFrame) -> pd.DataFrame:
    """One row per ICU stay: length, septic flag, demographics, first label hour."""
    grouped = hours.groupby(KEY, sort=False)
    stays = grouped.agg(
        hours=("ICULOS", "size"), septic=("SepsisLabel", "max"),
        age=("Age", "first"), gender=("Gender", "first"),
    ).reset_index()
    positive = hours[hours["SepsisLabel"] == 1]
    first = positive.groupby(KEY, sort=False)["ICULOS"].min().rename("first_positive_iculos")
    stays = stays.merge(first, on=KEY, how="left")
    stays["septic"] = stays["septic"].astype(bool)
    stays["age_group"] = pd.cut(stays["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    return stays


def missingness(hours: pd.DataFrame) -> pd.DataFrame:
    """Share of hourly rows with no measurement, per variable and hospital."""
    return hours.groupby("hospital")[DYNAMIC].apply(lambda g: g.isna().mean()).T[HOSPITALS]


def summary_table(hours: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for hospital in HOSPITALS:
        h = hours[hours["hospital"] == hospital]
        s = stays[stays["hospital"] == hospital]
        septic_ids = s.loc[s["septic"], "patient_id"]
        in_septic = h["patient_id"].isin(septic_ids)
        q1, median, q3 = s["hours"].quantile([0.25, 0.5, 0.75])
        rows[hospital] = {
            "ICU stays": len(s),
            "Hourly rows": len(h),
            "Septic stays": int(s["septic"].sum()),
            "Septic stays (%)": 100 * s["septic"].mean(),
            "Positive-label hours (%)": 100 * h["SepsisLabel"].mean(),
            "Stay length, median h (IQR)": f"{median:.0f} ({q1:.0f}-{q3:.0f})",
            "Septic stays positive from first hour (%)": 100 * _left_censored_share(h, s),
            "Age, median (years)": s["age"].median(),
            "Male (%)": 100 * (s["gender"] == 1).mean(),
            "Hours with SIRS >= 2, all (%)": 100 * h["sirs_alert"].mean(),
            "Hours with SIRS >= 2, septic stays (%)": 100 * h.loc[in_septic, "sirs_alert"].mean(),
            "Hours with SIRS >= 2, non-septic stays (%)": 100 * h.loc[~in_septic, "sirs_alert"].mean(),
            "HR missing (%)": 100 * h["HR"].isna().mean(),
            "Temp missing (%)": 100 * h["Temp"].isna().mean(),
            "WBC missing (%)": 100 * h["WBC"].isna().mean(),
            "Lactate missing (%)": 100 * h["Lactate"].isna().mean(),
        }
    return pd.DataFrame(rows)


def _left_censored_share(h, s):
    """Share of septic stays already labelled positive in their first row."""
    first_row = h.groupby("patient_id", sort=False)["ICULOS"].min()
    septic = s[s["septic"]].set_index("patient_id")
    if septic.empty:
        return float("nan")
    return float((septic["first_positive_iculos"] == first_row.reindex(septic.index)).mean())


# ---------------------------------------------------------------- figures

def _bar_pair(ax, labels, a_values, b_values, fmt="{:.1f}"):
    x = np.arange(len(labels))
    for offset, hospital, values in [(-0.2, "A", a_values), (0.2, "B", b_values)]:
        bars = ax.bar(x + offset, values, 0.4, label=LABELS[hospital], color=COLORS[hospital])
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def fig_cohort(stays, hours, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    by = stays.groupby("hospital")
    stay_pct = by["septic"].mean().reindex(HOSPITALS) * 100
    hour_pct = hours.groupby("hospital")["SepsisLabel"].mean().reindex(HOSPITALS) * 100
    _bar_pair(axes[0], ["Septic stays (%)", "Positive-label hours (%)"],
              [stay_pct["A"], hour_pct["A"]], [stay_pct["B"], hour_pct["B"]])
    axes[0].set_title("Label prevalence")
    counts = by.size().reindex(HOSPITALS)
    septic = by["septic"].sum().reindex(HOSPITALS)
    _bar_pair(axes[1], ["ICU stays", "Septic stays"],
              [counts["A"], septic["A"]], [counts["B"], septic["B"]], fmt="{:,.0f}")
    axes[1].set_title("Cohort size")
    fig.suptitle("1. Hospital A has a higher septic share than hospital B")
    _save(fig, path)


def fig_stay_length(stays, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    # Whole-hour data: integer-aligned 2 h bins; long tail collected at 100 h.
    bins = np.arange(0, 102, 2)
    for hospital in HOSPITALS:
        values = stays.loc[stays["hospital"] == hospital, "hours"]
        ax.hist(values.clip(upper=100), bins=bins, density=True, histtype="step", lw=1.8,
                color=COLORS[hospital], label=f"{LABELS[hospital]} (median {values.median():.0f} h)")
    ax.set_xlabel("Hours of ICU data per stay (> 100 h shown at 100)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("2. Stay length distribution")
    _save(fig, path)


def fig_demographics(stays, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    share = (stays.groupby("hospital")["age_group"].value_counts(normalize=True)
             .unstack().reindex(HOSPITALS)[AGE_LABELS] * 100)
    _bar_pair(axes[0], AGE_LABELS, share.loc["A"].to_numpy(), share.loc["B"].to_numpy())
    axes[0].set_xlabel("Age group (years)")
    axes[0].set_ylabel("% of ICU stays")
    axes[0].set_title("Age")
    male = stays.groupby("hospital")["gender"].apply(lambda g: (g == 1).mean() * 100).reindex(HOSPITALS)
    _bar_pair(axes[1], ["Male (%)"], [male["A"]], [male["B"]])
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Recorded gender")
    fig.suptitle("3. Age and gender (needed for the bias audit)")
    _save(fig, path)


def fig_missingness(missing, path):
    order = missing.mean(axis=1).sort_values().index
    m = missing.loc[order] * 100
    fig, ax = plt.subplots(figsize=(7, 9))
    y = np.arange(len(m))
    ax.barh(y - 0.2, m["A"], 0.4, color=COLORS["A"], label="Hospital A")
    ax.barh(y + 0.2, m["B"], 0.4, color=COLORS["B"], label="Hospital B")
    ax.set_yticks(y, m.index, fontsize=8)
    ax.set_xlabel("% of hourly rows with no measurement")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("4. Missing values per variable")
    _save(fig, path)


def fig_vitals(hours, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (column, (low, high)) in zip(axes.flat, VITAL_RANGES.items()):
        width = VITAL_BIN_WIDTH[column]
        bins = np.arange(low - width / 2, high + width, width)
        for hospital in HOSPITALS:
            values = hours.loc[hours["hospital"] == hospital, column].dropna()
            ax.hist(values.clip(low, high), bins=bins, density=True, histtype="step", lw=1.6,
                    color=COLORS[hospital], label=f"{LABELS[hospital]} (median {values.median():.1f})")
        ax.set_xlabel(f"{column} ({VITAL_UNITS[column]})")
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("5. Vital-sign distributions (measured hours only; extremes shown at the edges)")
    _save(fig, path)


def fig_sirs(hours, stays, path):
    septic_keys = stays.loc[stays["septic"], KEY]
    flag = hours[KEY].merge(septic_keys.assign(septic=True), on=KEY, how="left")["septic"].fillna(False).to_numpy()
    rates = hours.assign(septic=flag).groupby(["hospital", "septic"])["sirs_alert"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _bar_pair(ax, ["Non-septic stays", "Septic stays"],
              [rates.loc["A", False], rates.loc["A", True]], [rates.loc["B", False], rates.loc["B", True]])
    ax.set_ylabel("% of hours with SIRS >= 2")
    ax.set_title("6. How often the SIRS rule fires")
    _save(fig, path)


def fig_label_timing(stays, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 241, 6)
    for hospital in HOSPITALS:
        values = stays.loc[(stays["hospital"] == hospital) & stays["septic"], "first_positive_iculos"]
        ax.hist(values.clip(upper=240), bins=bins, density=True, histtype="step", lw=1.8,
                color=COLORS[hospital], label=f"{LABELS[hospital]} (median {values.median():.0f} h)")
    ax.set_xlabel("ICU hour of first positive label (septic stays; > 240 h shown at 240)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("7. When the label turns positive")
    _save(fig, path)


def _save(fig, path):
    fig.text(0.99, 0.005, f"Recreate: {RECREATE}", ha="right", va="bottom", fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def restrict_a_to_train(hours: pd.DataFrame, splits: Path) -> pd.DataFrame:
    """Keep hospital A's TRAINING stays only, plus all of B.

    Course rule: tune on validation, touch the test set once. Exploration can
    shape later choices, so A's calibration, validation and internal-test
    stays are left out. B is described only (nothing is tuned on it).
    """
    table = pd.read_csv(splits, dtype=str)
    train_ids = set(table.loc[(table["hospital"] == "A") & (table["split"] == "train"), "patient_id"])
    keep = (hours["hospital"] == "B") | hours["patient_id"].isin(train_ids)
    return hours[keep].reset_index(drop=True)


def run_comparison(data: Path, output: Path, splits: Path | None = None) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    hours = load_hours(data)
    LABELS["A"] = "Hospital A"
    if splits is not None:
        hours = restrict_a_to_train(hours, splits)
        LABELS["A"] = "Hospital A (train)"
    hours = add_sirs(hours)
    stays = stay_table(hours)
    missing = missingness(hours)
    summary = summary_table(hours, stays)
    figures = {
        "01_cohort_prevalence.png": lambda p: fig_cohort(stays, hours, p),
        "02_stay_length.png": lambda p: fig_stay_length(stays, p),
        "03_age_gender.png": lambda p: fig_demographics(stays, p),
        "04_missingness.png": lambda p: fig_missingness(missing, p),
        "05_vital_signs.png": lambda p: fig_vitals(hours, p),
        "06_sirs_firing_rate.png": lambda p: fig_sirs(hours, stays, p),
        "07_label_timing.png": lambda p: fig_label_timing(stays, p),
    }
    for name, draw in figures.items():
        draw(output / name)
    summary.to_csv(output / "hospital_summary.csv")
    (missing * 100).round(2).to_csv(output / "missingness_by_hospital.csv", header=["A_missing_pct", "B_missing_pct"])
    lines = [
        "# Hospital A vs B figures",
        "",
        "Descriptive only: nothing here is used to choose models or thresholds.",
        "Hospital A = its TRAINING split only (11,183 stays); A's calibration,",
        "validation and internal-test stays are left out. Hospital B = all 20,000",
        "stays, described only. All values are aggregates over stays or hourly rows.",
        "",
        "Recreate everything in this folder:",
        "",
        f"    {RECREATE}",
        "",
        "| File | Shows |",
        "|---|---|",
        "| 01_cohort_prevalence.png | Stays, septic stays, septic share and positive-hour share |",
        "| 02_stay_length.png | Stay length distribution (log scale) |",
        "| 03_age_gender.png | Age groups and recorded gender |",
        "| 04_missingness.png | % of hours with no measurement, per variable |",
        "| 05_vital_signs.png | HR, MAP, Temp, Resp distributions (measured hours) |",
        "| 06_sirs_firing_rate.png | % of hours with SIRS >= 2, septic vs non-septic stays |",
        "| 07_label_timing.png | ICU hour of the first positive label |",
        "| 08_sirs_threshold_tradeoff.png | SIRS at >= 1..4 criteria, validation only (made by "
        "`python -m sepsis_warning baseline --data data/physionet --output artifacts/baseline`) |",
        "| hospital_summary.csv | Summary table below |",
        "| missingness_by_hospital.csv | Missing % for all 34 hourly variables |",
        "",
        "## Summary table",
        "",
        _markdown_table(summary),
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _markdown_table(summary: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, str):
            return v
        return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 100 else f"{v:,.1f}"
    rows = ["| | Hospital A | Hospital B |", "|---|---:|---:|"]
    rows += [f"| {name} | {fmt(r['A'])} | {fmt(r['B'])} |" for name, r in summary.iterrows()]
    return "\n".join(rows)
