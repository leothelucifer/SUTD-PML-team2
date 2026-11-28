# Real-data ML workflow

This guide covers a leakage-aware local experiment on PhysioNet Challenge 2019.
Start with a small sample before downloading the full cohort. A successful run
shows the pipeline works on these records; it does not establish clinical
validity or generalisation to other hospitals.

## Set up and acquire data

Use Python 3.12 or newer. Keep downloaded records and all patient-level outputs
under the git-ignored `data/` and `artifacts/` directories.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[checks]"
```

The PhysioNet data is subject to its published
[license and attribution terms](https://physionet.org/content/challenge-2019/1.0.0/).
Do not redistribute the files.

```bash
# Start with a sample; use --limit 0 only when ready for the full cohort.
python scripts/download_data.py --limit 500 --seed 42 --output data/physionet
python -m sepsis_warning prepare \
  --data data/physionet --output data/physionet/combined.parquet
python -m sepsis_warning split \
  --data data/physionet/combined.parquet --output data/physionet/splits.csv
python -m sepsis_warning validate \
  --data data/physionet/combined.parquet \
  --splits data/physionet/splits.csv --output artifacts/data-checks
```

If the sample is too small to split, increase `--limit`. Preserve the seed and
split summary when comparing experiments. See [data versioning](../data-versioning.md)
before using DVC or restoring private data.

## Establish baselines and train

```bash
python -m sepsis_warning baseline \
  --data data/physionet/combined.parquet --output artifacts/baseline
python -m sepsis_warning tune \
  --data data/physionet/combined.parquet --output artifacts/tuning \
  --trials 3 --folds 3
python -m sepsis_warning train \
  --data data/physionet/combined.parquet --output artifacts/real \
  --models-config artifacts/tuning/tuned_models.toml \
  --splits-summary data/physionet/splits.csv.summary.json
python -m sepsis_warning model-report \
  --run artifacts/real --output artifacts/real-report \
  --tuning artifacts/tuning
```

Use a new empty output directory for each run. `train` fits and selects using
train, calibration, and validation data; it does not score either test split.
For experiment tracking, install the `tracking` extra and see
[MLflow tracking](../mlflow.md).

## Freeze and evaluate once

Record the selected model, feature configuration, and threshold before
evaluating held-out data. Then score the internal test and external hospital B
once:

```bash
python -m sepsis_warning evaluate \
  --data data/physionet/combined.parquet \
  --model artifacts/real/model.joblib \
  --output artifacts/evaluation
python -m sepsis_warning model-report \
  --run artifacts/real --output artifacts/real-report-evaluation \
  --tuning artifacts/tuning --evaluation artifacts/evaluation
```

Report the two test splits separately. Do not select or retune a model,
feature, or threshold using hospital B and still call it an untouched test.
Keep patient-level outputs private; only reviewed aggregate results belong in
reports.

## Next steps

- [Data profile](../data-profile.md) — field definitions and known cohort
  differences.
- [MLflow](../mlflow.md) — experiment lineage and registry promotion.
- [Monitoring](../monitoring-plan.md) — frozen references and drift checks.
- [Deploy to AWS](../deploy-to-any-account.md) — separate guide; AWS is not
  required for the local workflow.
