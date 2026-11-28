# Code map and ML invariants

For contributors changing the pipeline. The [root README](../README.md) is the
short project introduction; this page maps code to responsibilities and records
the evaluation rules that changes must preserve.

## Package map

The package uses a `src/` layout. Start at
[`cli.py`](../src/sepsis_warning/cli.py) for command routing, then follow the
owning module and its tests.

| Area | Code |
|---|---|
| Data ingestion and validation | [`data/`](../src/sepsis_warning/data): records, cohorts, schema, preparation, downloads, and saved splits |
| Features | [`features/`](../src/sepsis_warning/features): causal shared features, HMM, and signature features |
| Models and policy | [`models/`](../src/sepsis_warning/models): estimators, NumPy models, scoring, calibration helpers, and alert policy |
| Training | [`training/`](../src/sepsis_warning/training): experiment orchestration and patient-grouped tuning |
| Evaluation | [`evaluation/`](../src/sepsis_warning/evaluation): baselines, one-time frozen evaluation, and cohort analyses |
| Reports | [`reporting/`](../src/sepsis_warning/reporting): model reports, model cards, and ward replay output |
| Model lifecycle | [`lifecycle/`](../src/sepsis_warning/lifecycle): manifests, MLflow, registry, model loading, and promotion gates |
| Serving | [`serving/`](../src/sepsis_warning/serving): API, service, client, and alert store |
| Monitoring | [`monitoring/`](../src/sepsis_warning/monitoring): offline checks, live drift, telemetry, and Evidently integration |
| Cloud adapters | [`adapters/aws/`](../src/sepsis_warning/adapters/aws): Lambda, ECS job, SageMaker compatibility, and drift handler |
| Tests and helpers | [`tests/`](../tests) and [`scripts/`](../scripts) |

## Data and model flow

1. Read PhysioNet PSV records or generate an explicitly synthetic fixture.
2. Validate and optionally prepare records into Parquet; persist a patient
   split so later commands use the same cohort assignment.
3. Build causal features independently within each stay.
4. Fit candidate models on the training split, calibrate on the calibration
   split, and select the model and alert threshold on validation.
5. Freeze the selected bundle and record its data, feature, and code lineage.
6. Run [`evaluate`](../src/sepsis_warning/evaluation/frozen.py) once for the
   internal and external test splits. `train` does not score held-out tests.
7. Generate tables and figures from saved outputs with
   [`model-report`](../src/sepsis_warning/reporting/model_report.py).

The default split uses hospital A for train, calibration, validation, and
internal test; hospital B is reserved for external test. An ICU stay is the
split unit because the data has no person identifier.

## Invariants for ML changes

- The challenge `SepsisLabel` is already shifted six hours before its
  dataset-defined onset. Do not shift it again or describe the score as a
  validated six-hour clinical risk.
- At hour *t*, use only measurements available through *t* in that stay.
  Forward-fill only within the stay; never backfill or share patient state.
- Fit imputers, scalers, and other learned transforms on training data only.
- Keep labels, patient/hospital identifiers, administrative unit codes, and
  whole-stay summaries out of model inputs.
- Tune and select only on training, calibration, and validation splits. Freeze
  choices before `evaluate`; never tune against hospital B.
- Keep hourly threshold decisions (`hourly_positive`) distinct from
  cooldown-controlled notifications (`alert_event`).
- Treat first-row-positive stays as left-censored for inferred lead-time
  analysis. Synthetic fixtures validate software behavior, not clinical or
  cohort performance.

## Commands and checks

Run commands from the repository root after installing the editable package.
The CLI is discoverable with:

```bash
python -m sepsis_warning --help
```

Focused tests live under [`tests/`](../tests). Run the relevant test module
first; for shared behavior, run the suite:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
ruff check src tests scripts
```

The [CI/CD guide](cicd.md) lists the full automated quality gates. Keep raw
records, models, predictions, and experiment outputs in ignored `data/` and
`artifacts/` directories. Never commit them or load an untrusted Joblib bundle.
