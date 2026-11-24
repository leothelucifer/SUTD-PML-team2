# Run the whole project on ECS: runbook, costs and the stop button

Everything below runs in your own AWS account from GitHub. Nothing needs a SageMaker quota. Training, tracking,
approval, serving and monitoring are containers; what bills while idle can be switched off with one workflow.

## What runs where

| Part | Runs as | Bills while idle? | Started and stopped by |
|---|---|---|---|
| Training steps (prepare, tune, train, frozen evaluation, publish) | one-off Fargate tasks | no | the **pipeline** workflow |
| Promote | one-off Fargate task under the approver role | no | the **promote** workflow |
| Approve | Step Functions answer | no | the **approve** workflow |
| MLflow server and registry | Fargate service + RDS PostgreSQL | **yes** | the **power** workflow |
| Evidently UI | Fargate service | **yes** | the **power** workflow |
| Evidently drift report | one-off Fargate task | no | the **monitor** workflow |
| Scoring endpoint | Lambda container, IAM-signed function URL | no | **deploy** ("full deploy") |
| Predict demo | runs from GitHub against the endpoint | no | the **demo** workflow |

Feature flags in `infra/params/<environment>.json`: `TrainingBackend` (`ecs`), `MlflowBackend` (`ecs`),
`EnableSageMakerMlflowApp` (off when MLflow runs on ECS; switch it on and `MlflowBackend` back to `sagemaker` to use the
SageMaker MLflow App again), `EnableEvidently`.

## One-time setup

1. **Deploy** workflow, steps "stack and images only". It runs the stack update twice on purpose: the second run creates
   the MLflow and Evidently services once their image exists.
2. Check the quotas with **verify** (Fargate vCPU; see [ecs-training.md](ecs-training.md)).
3. The PhysioNet data is copied once with `aws_deploy.sh fetch-data` (the **pipeline** workflow does it when you pick
   `data_source = physionet`). A marker (`raw/physionet/_SUCCESS`) records the finished copy, and the prepared data
   carries its own marker with SHA-256 hashes, so reruns do nothing. Real data cannot be overwritten by a rehearsal.

## Demo sequence

| # | Do | Workflow | Notes |
|---|---|---|---|
| 1 | Start the tools | **power** start, component `all`, `allow_cidr` = your address as `x.x.x.x/32` | 5 to 10 minutes. The job summary prints the MLflow and Evidently URLs |
| 2 | Real-data run | **pipeline**: `data_source = physionet`, trials 6 | About 2 to 3 hours the first time. Stops at the approval |
| 3 | Review | open the MLflow URL and `s3://<bucket>/reports/<run>/` | metrics table, PR curves, figures |
| 4 | Promote | **promote** with the model version | Runs as the approver role; the GitHub user is recorded as reviewer |
| 5 | Approve | **approve** | Then the one frozen evaluation and the publish run |
| 6 | Serve | **deploy**, steps "full deploy" | Canary on Lambda |
| 7 | Predict | **demo**, `split = validation` | Replays stays through the live endpoint; the page goes to the private bucket |
| 8 | Monitor | **monitor** `reference` once, then `report` | Reference from hospital A validation; report from the prediction log |
| 9 | Stop | **power** stop, component `all` | Closes browser access, stops the services and the database |
| 10 | Check | **power** status | Only storage should be left |

Hospital B (`external_test`) is only replayed by **demo** for a model with a recorded frozen evaluation, and only after the
team freezes the model and policy decisions.

## Costs (estimates; the AWS pricing page is the authority)

| Item | While on | While off |
|---|---|---|
| MLflow service (0.5 vCPU, 2 GB) | about $0.04 per hour | $0 |
| Evidently UI (0.5 vCPU, 1 GB) | about $0.03 per hour | $0 |
| RDS `db.t4g.micro` | about $0.02 per hour | storage only (about $2.50 a month for 20 GB) |
| Public IPv4 on each running task | about $0.005 per hour | $0 |
| A training task (4 vCPU, 16 GB) | about $0.30 per hour | $0 |
| Lambda scoring | per request | $0 |
| KMS key, secrets, S3, ECR, logs | a few dollars a month | same |

All tools on is about $0.10 an hour, so an 8-hour demo day costs about a dollar plus the training run. Left on for a
month it would be about $75, which is why the **power** workflow exists. It never runs by itself: nothing here is on a
schedule, so remember to stop. The **power** `status` action prints what is on and the estimated hourly cost.

AWS starts a stopped RDS database again after 7 days. If the environment sits idle for a week, run **power** stop again.

## Access and safety

- The tools have no login by design (course project). Their security groups accept nobody until `allow_cidr` opens one
  address for that start; **stop** closes it again. `0.0.0.0/0` needs the explicit `allow_anywhere` box.
- MLflow holds metrics, hashes and model files, and Evidently holds aggregates: no stay identifiers and no measurement
  values. The ward replay page from the **demo** workflow does hold PhysioNet trajectories, so it stays in the private bucket.
- Promotion runs under a different AWS role from training, and the GitHub environment's required reviewers (with
  "Prevent self-review") are the human approval. The next step refuses a model that was never promoted before it scores
  the test splits.

## If something fails

| Symptom | Look here |
|---|---|
| Pipeline says MLflow is stopped | **power** start |
| A step fails | the failure reason is in `s3://<bucket>/training-output/<run>/<step>/output/failure`; logs in `/ecs/<environment>-training` |
| `stack` cannot create the MLflow service | the tools image must exist first: run **deploy** again |
| Evidently report has no reference | **monitor** with action `reference` for the deployed model |
| Task cannot start | Fargate vCPU quota (**verify**), or lower `EcsTrainingCpu` |
