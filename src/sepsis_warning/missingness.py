"""Aggregate missingness audit and exploratory train/validation baselines."""

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DYNAMIC, STATIC, Patient, split_patients

SPLITS = ("train", "validation")
STRATEGIES = (
    "current_median",
    "current_median_missing",
    "ffill_median",
    "ffill_median_missing",
    "ffill_median_missing_age",
)


def _record_arrays(
    patient: Patient,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = patient.frame
    measurements = frame.loc[:, DYNAMIC].to_numpy(dtype=np.float64, na_value=np.nan)
    hours = frame["ICULOS"].to_numpy(dtype=np.float64)
    context = frame.loc[:, STATIC].to_numpy(dtype=np.float64, na_value=np.nan)
    labels = frame["SepsisLabel"].to_numpy(dtype=np.int8)
    return measurements, hours, context, labels


def record_features(
    patient: Patient,
    medians: np.ndarray,
    strategy: str,
    active_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build features using only current/past values from this patient."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    raw, hours, context, labels = _record_arrays(patient)
    raw = raw[:, active_columns]
    observed = np.isfinite(raw)
    current = np.where(observed, raw, medians)
    blocks = [current]

    if strategy.startswith("ffill"):
        carried = pd.DataFrame(raw).ffill().to_numpy(dtype=np.float64)
        blocks = [np.where(np.isfinite(carried), carried, medians)]

    if strategy in {
        "current_median_missing",
        "ffill_median_missing",
        "ffill_median_missing_age",
    }:
        blocks.append((~observed).astype(np.float64))

    if strategy == "ffill_median_missing_age":
        last_seen = np.maximum.accumulate(
            np.where(observed, hours[:, None], -np.inf), axis=0
        )
        never_seen = ~np.isfinite(last_seen)
        age = np.where(
            never_seen,
            (hours - hours[0] + 1)[:, None],
            hours[:, None] - last_seen,
        )
        blocks.extend((never_seen.astype(np.float64), age))

    return np.column_stack((*blocks, context)).astype(np.float32), labels


def _aggregate_missingness(
    patients: list[Patient], split: str
) -> tuple[list[dict], list[dict], list[dict]]:
    variable_totals = np.zeros(len(DYNAMIC), dtype=np.int64)
    variable_rows = 0
    record_missingness: dict[str, list[float]] = defaultdict(list)
    hourly_counts: dict[int, np.ndarray] = {}
    hourly_missing: dict[int, np.ndarray] = {}
    hourly_records: dict[int, int] = defaultdict(int)

    for patient in patients:
        values = patient.frame.loc[:, DYNAMIC].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        missing = ~np.isfinite(values)
        variable_totals += missing.sum(axis=0)
        variable_rows += len(values)
        record_missingness["positive" if patient.septic else "no_positive"].append(
            missing.mean(axis=0)
        )

        hours = patient.frame["ICULOS"].to_numpy(dtype=np.int64)
        for hour in np.unique(hours):
            mask = hours == hour
            if hour not in hourly_counts:
                hourly_counts[hour] = np.zeros(len(DYNAMIC), dtype=np.int64)
                hourly_missing[hour] = np.zeros(len(DYNAMIC), dtype=np.int64)
            hourly_counts[hour] += mask.sum()
            hourly_missing[hour] += missing[mask].sum(axis=0)
            hourly_records[hour] += 1

    variable_rows_out = []
    for index, variable in enumerate(DYNAMIC):
        record_rates = [
            rates[index] for rates in record_missingness["positive"]
            + record_missingness["no_positive"]
        ]
        variable_rows_out.append({
            "split": split,
            "variable": variable,
            "records": len(patients),
            "hourly_rows": variable_rows,
            "missing_hours": int(variable_totals[index]),
            "missing_fraction": float(variable_totals[index] / variable_rows),
            "mean_record_missing_fraction": float(np.mean(record_rates)),
            "never_observed_records": int(sum(rate == 1 for rate in record_rates)),
        })

    group_rows = []
    for group, rates in record_missingness.items():
        if not rates:
            continue
        mean_rates = np.mean(rates, axis=0)
        for index, variable in enumerate(DYNAMIC):
            group_rows.append({
                "split": split,
                "label_group": group,
                "records": len(rates),
                "variable": variable,
                "mean_record_missing_fraction": float(mean_rates[index]),
            })

    hourly_rows = []
    for hour in sorted(hourly_counts):
        for index, variable in enumerate(DYNAMIC):
            hourly_rows.append({
                "split": split,
                "icu_hour": hour,
                "records_present": hourly_records[hour],
                "variable": variable,
                "missing_fraction": float(
                    hourly_missing[hour][index] / hourly_counts[hour][index]
                ),
            })
    return variable_rows_out, group_rows, hourly_rows


def _training_medians(patients: list[Patient]) -> tuple[np.ndarray, np.ndarray]:
    observations = pd.concat(
        [patient.frame.loc[:, DYNAMIC] for patient in patients],
        ignore_index=True,
    )
    medians = observations.median(axis=0).to_numpy(dtype=np.float64)
    active = np.isfinite(medians)
    if not active.any():
        raise ValueError("Training records contain no observed measurement values")
    return medians[active], active


def _model_matrix(
    patients: list[Patient],
    medians: np.ndarray,
    strategy: str,
    active_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for patient in patients:
        matrix, target = record_features(
            patient, medians, strategy, active_columns
        )
        matrices.append(matrix)
        labels.append(target)
    return np.concatenate(matrices), np.concatenate(labels)


def _compare_strategies(
    training: list[Patient], validation: list[Patient]
) -> list[dict]:
    medians, active = _training_medians(training)
    comparisons = []
    for strategy in STRATEGIES:
        train_x, train_y = _model_matrix(training, medians, strategy, active)
        validation_x, validation_y = _model_matrix(
            validation, medians, strategy, active
        )
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=500, solver="lbfgs"),
        )
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(validation_x)[:, 1]
        comparisons.append({
            "strategy": strategy,
            "positive_hour_fraction": float(validation_y.mean()),
            "average_precision": float(
                average_precision_score(validation_y, probabilities)
            ),
            "auroc": float(roc_auc_score(validation_y, probabilities)),
            "brier_score": float(brier_score_loss(validation_y, probabilities)),
        })
    return comparisons


def _bar_chart(title: str, labels: list[str], values: list[float], maximum: float) -> str:
    width, row_height, left, right = 920, 24, 230, 70
    height = 72 + row_height * len(labels)
    bars = []
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 44 + index * row_height
        length = max(0, (width - left - right) * value / maximum)
        bars.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end">'
            f'{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{length:.2f}" height="15" '
            'fill="#3b82a0"/>'
            f'<text x="{left + length + 5:.2f}" y="{y + 13}">'
            f'{value:.3f}</text>'
        )
    return (
        f'<svg role="img" aria-label="{html.escape(title)}" '
        f'viewBox="0 0 {width} {height}" width="100%" height="{height}">'
        f'<text x="{left}" y="22" font-size="16" font-weight="bold">'
        f'{html.escape(title)}</text>{"".join(bars)}</svg>'
    )


def _hourly_chart(hourly_rows: list[dict]) -> str:
    selected = ("HR", "O2Sat", "Temp", "MAP", "Lactate")
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706")
    hours = sorted({
        row["icu_hour"] for row in hourly_rows
        if row["split"] == "train" and row["icu_hour"] <= 48
    })
    if not hours:
        return ""
    width, height, left, top = 920, 300, 58, 38
    plot_width, plot_height = width - left - 120, height - 80
    series = []
    legend = []
    for variable, color in zip(selected, colors, strict=True):
        rates = {
            row["icu_hour"]: row["missing_fraction"]
            for row in hourly_rows
            if row["split"] == "train" and row["variable"] == variable
            and row["icu_hour"] in hours
        }
        points = [
            (
                left + (hour - hours[0]) / max(hours[-1] - hours[0], 1) * plot_width,
                top + (1 - rates[hour]) * plot_height,
            )
            for hour in hours if hour in rates
        ]
        if points:
            series.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}"/>'
            )
            legend.append(
                f'<text x="{left + plot_width + 12}" y="{top + 16 * len(legend)}" '
                f'fill="{color}">{html.escape(variable)}</text>'
            )
    axes = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#555"/>'
        f'<line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#555"/>'
        f'<text x="{left}" y="{height - 20}">ICU hour (first 48)</text>'
        f'<text x="8" y="{top}" transform="rotate(-90 8 {top})">'
        'Missing fraction</text>'
        f'<text x="{left}" y="22" font-size="16" font-weight="bold">'
        'Training missingness by ICU hour</text>'
    )
    return (
        f'<svg role="img" aria-label="Training missingness by ICU hour" '
        f'viewBox="0 0 {width} {height}" width="100%" height="{height}">'
        f'{axes}{"".join(series)}{"".join(legend)}</svg>'
    )


def _write_report(
    output: Path,
    variable_rows: list[dict],
    group_rows: list[dict],
    hourly_rows: list[dict],
    metrics: list[dict],
) -> None:
    training = [row for row in variable_rows if row["split"] == "train"]
    top = sorted(training, key=lambda row: row["missing_fraction"], reverse=True)[:15]
    group_data = [
        row for row in group_rows
        if row["split"] == "train" and row["variable"] in {
            entry["variable"] for entry in top
        }
    ]
    group_bars = [
        row for row in group_data
        if row["label_group"] in {"positive", "no_positive"}
    ]
    metrics_sorted = sorted(metrics, key=lambda row: row["average_precision"], reverse=True)
    charts = [
        _bar_chart(
            "Training hourly missing fraction (top 15)",
            [row["variable"] for row in top],
            [row["missing_fraction"] for row in top],
            1.0,
        ),
        _bar_chart(
            "Training record missingness by whole-record label group",
            [
                f'{row["variable"]} ({row["label_group"].replace("_", " ")})'
                for row in group_bars
            ],
            [row["mean_record_missing_fraction"] for row in group_bars],
            1.0,
        ),
        _hourly_chart(hourly_rows),
        _bar_chart(
            "Validation average precision by fixed strategy",
            [row["strategy"] for row in metrics_sorted],
            [row["average_precision"] for row in metrics_sorted],
            max(row["average_precision"] for row in metrics_sorted) or 1.0,
        ),
    ]
    (output / "missingness_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Missingness audit</title><style>"
        "body{font:15px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}"
        "svg{margin:1rem 0;border:1px solid #ddd} "
        "text{font:12px system-ui}</style></head><body>"
        "<h1>Training and validation missingness audit</h1>"
        "<p>Label-group comparisons are exploratory, retrospective associations; "
        "they do not show that missingness indicates sepsis. Validation is used "
        "for validation summaries and the fixed strategy comparison, not for "
        "fitting medians or models. Internal and external test cohorts are "
        "excluded.</p>"
        + "".join(charts)
        + "</body></html>\n",
        encoding="utf-8",
    )


def run_missingness_analysis(
    patients: list[Patient],
    output: Path,
    *,
    seed: int = 42,
    synthetic: bool = False,
) -> dict:
    """Write aggregate-only summaries and fixed validation comparisons."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    splits = split_patients(patients, seed)
    training, validation = (splits[name] for name in SPLITS)
    variable_rows = []
    group_rows = []
    hourly_rows = []
    for name, cohort in zip(SPLITS, (training, validation), strict=True):
        variables, groups, hours = _aggregate_missingness(cohort, name)
        variable_rows.extend(variables)
        group_rows.extend(groups)
        hourly_rows.extend(hours)
    metrics = _compare_strategies(training, validation)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(variable_rows).to_csv(output / "missingness_by_variable.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "missingness_by_record_group.csv", index=False)
    pd.DataFrame(hourly_rows).to_csv(output / "missingness_by_icu_hour.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "validation_strategy_metrics.csv", index=False)
    summary = {
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "seed": seed,
        "used_splits": list(SPLITS),
        "training_records": len(training),
        "validation_records": len(validation),
        "strategies": metrics,
        "limitations": [
            "Synthetic-fixture scores are software checks, not clinical or real-cohort evidence."
            if synthetic else
            "Scores are exploratory and do not establish clinical performance.",
            "Label-group missingness is a retrospective association, not evidence that missingness indicates sepsis.",
            "The supplied benchmark label is already shifted six hours early.",
            "Hourly metrics do not account for correlation within patients.",
            "No strategy or alert threshold is selected from these results.",
            "Internal-test and external-test cohorts are not used.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(output, variable_rows, group_rows, hourly_rows, metrics)
    return summary
