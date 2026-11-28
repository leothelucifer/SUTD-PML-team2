# Documentation index

The root [README](../README.md) is the short project introduction and quick
start. Use this page to jump directly to a workflow or technical reference;
most visitors do not need to read every document.

## Start here

- [Code map and research invariants](agent-repository-guide.md) — package
  structure, data flow, and rules for leakage-safe ML changes.
- [Real-data workflow](testing/real-data.md) — staged download, preparation,
  validation, training, and frozen evaluation.
- [AWS API performance test](testing/performance.md) — bounded Locust load on
  the deployed API using temporary, anonymized Hospital B payloads.
- [Data profile](data-profile.md) — cohort layout, labels, fields, missingness,
  and known source differences.

## Machine learning and data engineering

- [Data versioning](data-versioning.md) — DVC setup and private data restore.
- [Clinical-use design and validation plan](clinical-use-design-and-validation-plan.md) —
  intended use, evaluation evidence, and limits.
- [Hospital A vs B](hospital-a-vs-b.md) and [SIRS failure patterns](sirs-failure-patterns.md) —
  descriptive cohort and baseline analyses.
- [H03 ablation plan](h03-ablation-plan.md) — feature ablation question and
  experiment protocol.
- [MLflow tracking and model registry](mlflow.md) — runs, promotion, and model
  lineage.
- [Monitoring plan](monitoring-plan.md) — reference data, drift checks, and
  alert interpretation.

## Serving and operations

- [Serving](serving.md) — local API, Docker image, latency, and request contract.
- [Ward replay](ward-replay.md) — offline replay through the scoring service.
- [Production-like local stack](local-production.md) — Docker Compose release,
  health checks, and rollback.
- [Runbooks](runbooks/README.md) — drift alarms, scoring errors, and model
  rollback.

## CI/CD and AWS

- [CI/CD overview](cicd.md) — change-based checks, deployment planning, and
  environment concurrency.
- [GitHub workflow reference](WORKFLOWS.md) — manual workflow inputs and
  effects.
- [AWS run order](RUN-ORDER.md) — deployment sequence for an AWS environment.
- [Deploy to an AWS account](deploy-to-any-account.md) — one-time account
  setup and deployment.
- [ECS runbook](ecs-runbook.md) — current AWS components, costs, and recovery.
- [CloudFormation guide](../infra/cloudformation/README.md) — infrastructure
  layout and local template checks.
- [IAM review](iam-review.md) and [AWS assumptions](aws-assumptions.md) —
  permissions and recorded AWS behavior.

## Reports and reference material

- [Final report draft](../reports/final/report.md).
- [Final validation record](../reports/final-validation-2026-11-26/report.md) —
  a dated evidence snapshot; some supporting run artifacts were local and are
  not committed.
- [Presentation source and rebuild instructions](presentation/README.md).
- [Risk register](risk-register.md) — dated project risk review; owner/status
  fields are a snapshot, not live issue tracking.
- [Architecture review](architecture-review-and-design.md) — dated design
  review, not the current implementation specification.
- [Legacy SageMaker design](aws-serverless.md) — retained for the optional
  backend only; the current default uses ECS Fargate.

The project task tracker and contribution record are kept at the repository
root for course administration; they are not required to run the software.
