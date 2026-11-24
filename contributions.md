# Team members and contributions

Team 2, 61.501 Production Ready ML: early sepsis warning on the PhysioNet 2019 challenge data.

This is the contribution statement the project brief asks for. It summarises
[TEAM-TASKS.md](TEAM-TASKS.md), which holds every task's evidence, review history and
acceptance criteria. Live status is in the GitHub issues linked from
[team-status.md](team-status.md). Each status below is the owner's own record in the
tracker, last updated between 2026-11-08 and 2026-11-16. Where a tracker note no longer
matches the repository, the difference is listed under
[Checked against the repository](#checked-against-the-repository-on-2026-11-24).

Before submission: add full names and student IDs (the repository records first names
only), and have each person confirm their section.

## Summary

| Member | Area | Done | In review | In progress | Not started or deferred | Slides |
|---|---|---|---|---|---|---|
| Edwin | Data and baselines | E01, E04, E06 | E02, E03 | E05 | E07 | 4, 5, 9, 10 |
| Jawad | Features and the utility metric | J01 to J06 | | J07 | | 6, 7 |
| Kaiyuan | Models, calibration and alert policy | | K01 to K06 | K07 | | 3, 8, 11, 12 |
| Josh | Temporal feature experiments | H01, H02, H03, H06 | | | H04 (optional), H05 (deferred) | 14 |
| leo | Reproducibility, serving and deployment design | L04 | L01, L02, L03, L05 | L06 | L07 | 16, 17, 18 |
| Mashie | Replay, monitoring, risk and reporting | | M02, M05 | M01, M04 | M03, M06, M07 | 13, 15, 19, 20 |

Slides 1, 2, 21 and 22 (cover, problem, lessons, close) are shared and assembled by Mashie (M06).

## Edwin: data and baselines

- **E01 Download and Parquet preparation (done).** `prepare` command
  ([prepare.py](src/sepsis_warning/prepare.py), [test_prepare.py](tests/test_prepare.py)). Full data:
  40,336 records and 1,552,210 hourly rows, with no hash mismatch against the manifest.
- **E02 Data versioning with DVC (in review).** [dvc.yaml](dvc.yaml), [dvc.lock](dvc.lock),
  [docs/data-versioning.md](docs/data-versioning.md). A fresh clone restored the prepared files with matching hashes.
- **E03 Schema checks with Pandera (in review).** `validate` command ([schema.py](src/sepsis_warning/schema.py),
  [test_schema.py](tests/test_schema.py)). All stays pass the hard rules; extreme values are counted for review, never rejected.
- **E04 Hospital A and B split (done).** `split` command ([splits.py](src/sepsis_warning/splits.py),
  [test_splits.py](tests/test_splits.py)): 55/15/15/15 of hospital A, all of hospital B as the external test, seed 42.
- **E05 Exploratory analysis (in progress).** `compare-hospitals` command
  ([hospital_comparison.py](src/sepsis_warning/hospital_comparison.py)), seven figures in
  [reports/figures](reports/figures/README.md), findings in [docs/hospital-a-vs-b.md](docs/hospital-a-vs-b.md).
- **E06 SIRS rule baseline (done).** `baseline` command ([baselines.py](src/sepsis_warning/baselines.py),
  [test_baselines.py](tests/test_baselines.py)), results in [reports/baselines](reports/baselines), failure patterns in
  [docs/sirs-failure-patterns.md](docs/sirs-failure-patterns.md). The utility was recomputed independently with the official formula.
- **E07 Slides 4, 5, 9, 10 (not started).**

## Jawad: features and the utility metric

- **J01 to J03 Causal features (done).** The shared builder in [features.py](src/sepsis_warning/features.py): last value,
  missing now, never seen, hours since seen, and a trailing window of mean, standard deviation, count and change.
  Forward fill stays inside one stay and never uses a later hour.
- **J04 Leakage tests (done).** [test_invariants.py](tests/test_invariants.py): corrupting the next hour leaves earlier
  features unchanged, scoring a prefix matches the full stay, and a deliberately leaking fill is caught.
- **J05 Utility score (done).** `normalized_utility` in [policy.py](src/sepsis_warning/policy.py) matches the published
  example and is timed on 24,000 patient-hours.
- **J06 Shared feature module (done).** `train`, replay and the scoring service call the same `build_features`; the
  saved column list travels with the model. [scripts/causal_features.py](scripts/causal_features.py) demonstrates it.
- **J07 Slides 6 and 7 (in progress).** Drafted; team review not recorded.

## Kaiyuan: models, calibration and alert policy

- **K01 to K03 Hand-written NumPy models (in review).** Logistic regression
  ([logistic.py](src/sepsis_warning/logistic.py)), random forest ([random_forest.py](src/sepsis_warning/random_forest.py))
  and histogram gradient boosting ([gradient_boosting.py](src/sepsis_warning/gradient_boosting.py),
  [histogram_trees.py](src/sepsis_warning/histogram_trees.py)), each checked against scikit-learn in
  [test_model_config.py](tests/test_model_config.py). They are optional candidates (`train --include-numpy-models`).
  K03 is a gradient-boosting re-implementation, not LightGBM; the owner is to confirm that this meets the item.
- **K04 Probability calibration (in review).** Platt scaling on a frozen model, fitted on the calibration split
  (`fit_calibrated` in [modeling.py](src/sepsis_warning/modeling.py)).
- **K05 Alert threshold and six-hour cooldown (in review).** Threshold tuned to the false-alert budget on validation,
  with the one-standard-error selection rule and bootstrap intervals ([policy.py](src/sepsis_warning/policy.py),
  [test_utility_weights.py](tests/test_utility_weights.py)).
- **K06 Fairness checks (in review).** Subgroup utility gaps by recorded sex and age band in
  [monitoring.py](src/sepsis_warning/monitoring.py). Age bands are <45, 45-64, 65-79 and 80+.
- **K07 Slides 3, 8, 11, 12 (in progress).**

## Josh: temporal feature experiments

- **H01 HMM forward-pass states (done).** [hmm.py](src/sepsis_warning/hmm.py) (forward filter and Baum-Welch),
  [scripts/fit_hmm.py](scripts/fit_hmm.py).
- **H02 Signature features (done).** [signature.py](src/sepsis_warning/signature.py): depth-2 log-signature over a
  trailing six-hour window.
- **H03 Ablation with and without the new features (done).** [scripts/h03_ablation.py](scripts/h03_ablation.py),
  plan in [docs/h03-ablation-plan.md](docs/h03-ablation-plan.md). On validation (3,051 stays) neither feature set
  helps logistic regression or gradient boosting: every 95% paired-bootstrap interval includes zero. Random forest
  with HMM states gains 0.012 average precision (95% interval 0.003 to 0.024), the only one of nine intervals above zero.
- **H04 Optional GMM (not started). H05 Optional GRU (deferred):** deep learning is outside the course rules.
- **H06 Slide 14 (done).** [latent.html](docs/presentation/source/slides/latent.html); the `.pptx` is not rebuilt.

## leo: reproducibility, serving and deployment design

- **L01 MLflow tracking and registry (in review).** [tracking.py](src/sepsis_warning/tracking.py),
  [registry.py](src/sepsis_warning/registry.py), [docs/mlflow.md](docs/mlflow.md). `train --register-model` registers a
  candidate with its hashes; `promote` moves the production alias after a named review; `serve` checks the hash first.
- **L02 FastAPI service (in review).** [service.py](src/sepsis_warning/service.py), [api.py](src/sepsis_warning/api.py):
  `/health`, `/model`, `/score`, `/predict`. Tested for replay parity, prefix scoring and concurrent isolation.
- **L03 Docker image (in review).** [Dockerfile](Dockerfile), [scripts/docker_smoke_test.sh](scripts/docker_smoke_test.sh),
  [docs/serving.md](docs/serving.md): non-root user, pinned requirements, no data or model in the image.
- **L04 CI with GitHub Actions (done).** [tests.yml](.github/workflows/tests.yml).
- **L05 Latency tests (in review).** [scripts/latency.py](scripts/latency.py). In the Docker image on 2 CPUs, one client:
  p50 41 ms, p95 67 ms, p99 116 ms for a 24-hour stay. No numeric target is agreed yet.
- **L06 AWS deployment design (in progress).** CloudFormation templates in
  [infra/cloudformation](infra/cloudformation/README.md), checked with cfn-lint in CI. The tracker records that
  nothing is deployed to an AWS account.
- **L07 Slides 16, 17, 18 (not started).**

## Mashie: replay, monitoring, risk and reporting

- **M01 Hospital B replay (in progress).** [cross_source.py](src/sepsis_warning/cross_source.py) and
  `monitor-replay --split external_test`, with a stay-level bootstrap ([test_hospital_b_replay.py](tests/test_hospital_b_replay.py)).
  Verified on synthetic data only: the real hospital B replay waits for the team to freeze the model.
- **M02 Two-speed drift monitoring (in review).** `monitor-setup` and `monitor-replay`
  ([monitoring.py](src/sepsis_warning/monitoring.py), [test_monitoring.py](tests/test_monitoring.py)); plan, owners and
  runbook in [docs/monitoring-plan.md](docs/monitoring-plan.md).
- **M03 SHAP explanations (not started).**
- **M04 Model card (in progress).** [model_card.py](src/sepsis_warning/model_card.py) builds the selection card served
  by `GET /model`. There is no written model card document yet.
- **M05 Risk register (in review).** [docs/risk-register.md](docs/risk-register.md): 25 risks with likelihood, impact,
  owner, evidence and remaining exposure.
- **M06 Final deck assembly (not started). M07 Slides 13, 15, 19, 20 (not started).**

## Checked against the repository on 2026-11-24

These tracker notes are out of date. The tracker rows themselves are unchanged; their owners should update them.

- **K04:** the tracker says the Brier-score promotion gate is only on a branch. [gates.py](src/sepsis_warning/gates.py)
  is in this repository.
- **L06:** the tracker says canary releases, `pipeline.yaml`, `observability.yaml` and `apigw.yaml` are only on a
  branch. All four are in this repository.
- **L04:** `tests.yml` now does more than the tracker lists: Ruff lint, mypy on the serving modules, `pip-audit`, a
  coverage floor, Docker image builds with a smoke test and a latency budget, and CloudFormation lint.
- **K06:** the tracker says no fairness report is recorded. `model-report` now writes a subgroup table by sex, age
  band and hospital with intervals. No table from real data is committed yet.
- **M03:** SHAP is still not started. `train` and `model-report` now give grouped permutation importance instead.

## Work the tracker does not attribute

These parts of the repository have no task in the tracker. Record who built each one so the report's contribution
table is complete.

- Hyperparameter search: `tune` ([tuning.py](src/sepsis_warning/tuning.py)).
- The one frozen test evaluation: `evaluate` ([evaluation.py](src/sepsis_warning/evaluation.py)).
- Results tables and figures: `model-report` ([model_report.py](src/sepsis_warning/model_report.py),
  [importance.py](src/sepsis_warning/importance.py)).
- ICU ward replay: `ward` ([ward.py](src/sepsis_warning/ward.py), [docs/ward-replay.md](docs/ward-replay.md)).
- The report draft: [reports/final/report.md](reports/final/report.md).
