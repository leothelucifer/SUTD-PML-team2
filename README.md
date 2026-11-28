# Early sepsis warning across hospitals

A research pipeline for hourly sepsis-warning experiments on the PhysioNet/CinC
Challenge 2019 data. It uses classical machine-learning models, causal
patient-level features, an explicit alert policy, and a held-out source cohort
for external evaluation.

**Research software only:** scores are not diagnoses or treatment advice.
`SepsisLabel` is already shifted six hours before the challenge-defined onset;
the model estimates that benchmark label, not a clinically validated six-hour
risk.

## Quick start

Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m sepsis_warning demo
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

The demo generates synthetic data and writes to `artifacts/demo/`. It needs no
PhysioNet download or AWS account. Choose a new output directory if that path
already exists. Synthetic results validate software behavior only; they are
not evidence of model performance.

## Find the right guide

| Goal | Start here |
|---|---|
| Understand the repository and code | [Code map and data contracts](docs/agent-repository-guide.md) |
| Run a real-data ML experiment | [Real-data workflow](docs/testing/real-data.md) |
| Measure the deployed API | [AWS API performance test](docs/testing/performance.md) |
| Understand the dataset and evaluation boundaries | [Data profile](docs/data-profile.md), [validation plan](docs/clinical-use-design-and-validation-plan.md) |
| Version private data | [Data versioning](docs/data-versioning.md) |
| Track experiments and review models | [MLflow and registry](docs/mlflow.md) |
| Run the scoring API or local Docker stack | [Serving](docs/serving.md), [local production-like stack](docs/local-production.md) |
| Review monitoring and runbooks | [Monitoring plan](docs/monitoring-plan.md), [runbooks](docs/runbooks/README.md) |
| Understand CI/CD or operate AWS workflows | [Start with the CI/CD guide](docs/cicd.md) |
| Set up AWS or inspect infrastructure | [Deploy to an AWS account](docs/deploy-to-any-account.md), [CloudFormation](infra/cloudformation/README.md) |
| Browse all documentation | [Documentation index](docs/README.md) |

## Repository map

| Path | Responsibility |
|---|---|
| [`src/sepsis_warning/data`](src/sepsis_warning/data) | Read and validate records, prepare Parquet, download data, and persist patient splits |
| [`src/sepsis_warning/features`](src/sepsis_warning/features) | Causal, patient-local feature construction |
| [`src/sepsis_warning/models`](src/sepsis_warning/models) | Candidate models, probability scoring, and alert policy |
| [`src/sepsis_warning/training`](src/sepsis_warning/training) | Training and grouped hyperparameter search |
| [`src/sepsis_warning/evaluation`](src/sepsis_warning/evaluation) | Baselines, frozen evaluation, and cohort analyses |
| [`src/sepsis_warning/lifecycle`](src/sepsis_warning/lifecycle) | MLflow tracking, registry, model manifests, and promotion gates |
| [`src/sepsis_warning/serving`](src/sepsis_warning/serving) | Scoring API, client, and alert persistence |
| [`src/sepsis_warning/monitoring`](src/sepsis_warning/monitoring) | Prediction logs and drift-monitoring checks |
| [`scripts`](scripts) / [`tests`](tests) | Data and CI helpers / unit, contract, and regression tests |

The command-line interface is [`src/sepsis_warning/cli.py`](src/sepsis_warning/cli.py);
run `python -m sepsis_warning --help` for the current command list.

## Data and result safety

Keep downloaded patient records, model bundles, and patient-level predictions
under the ignored `data/` and `artifacts/` directories. Never commit or share
them. Freeze features, calibration, model, and alert policy before calling
`evaluate` on either test split; do not retune against hospital B and describe
it as an untouched test.

The detailed project report is a draft at
[`reports/final/report.md`](reports/final/report.md). Consult the documentation
index for experiment design, infrastructure, operational details, and archived
project records.
