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
| Scoring endpoint | Lambda container, IAM-signed function URL | no | **deploy** (mode `auto` or "full deploy") |
| Predict demo | runs from GitHub against the endpoint | no | the **demo** workflow |

**No SageMaker access needed.** `infra/params/dev.json` has `EnableSageMaker=false`: the stack then creates no SageMaker
resource or permission (no training role, no MLflow App, no SageMaker statements in the team policies), and no script calls
a SageMaker API (**verify** skips its SageMaker checks, **deploy** reads the published model from S3 instead of the
SageMaker MLflow App). When AWS approves the access, set `EnableSageMaker` to `true` and, if you want to use it,
`TrainingBackend` and `MlflowBackend` to `sagemaker` (and `EnableSageMakerMlflowApp` to `true`) in the params file, then
run **deploy**. A rule refuses a SageMaker backend while `EnableSageMaker` is `false`.

Feature flags in `infra/params/<environment>.json`: `TrainingBackend` (`ecs`), `MlflowBackend` (`ecs`),
`EnableSageMakerMlflowApp` (off when MLflow runs on ECS; switch it on and `MlflowBackend` back to `sagemaker` to use the
SageMaker MLflow App again), `EnableEvidently`.

## One-time setup

1. **Deploy** workflow, mode "stack and images only" (or `auto`, which does the same on a new account). It runs the stack update twice on purpose: the second run creates
   the MLflow and Evidently services once their image exists.
2. Check the quotas with **verify** (Fargate vCPU; the deploy workflow reports the required capacity).
3. The PhysioNet data is copied once with `aws_deploy.sh fetch-data` (the **pipeline** workflow does it when you pick
   `data_source = physionet`). A marker (`raw/physionet/_SUCCESS`) records the finished copy, and the prepared data
   carries its own marker with SHA-256 hashes, so reruns do nothing. Real data cannot be overwritten by a rehearsal.

## Demo sequence

| # | Do | Workflow | Notes |
|---|---|---|---|
| 1 | Start the tools | **power** start, component `all`, `allow_cidr` = your address as `x.x.x.x/32` | 5 to 10 minutes. The job summary prints the MLflow and Evidently URLs |
| 2 | Real-data run | **pipeline**: `action = start`, `data_source = physionet`, trials 6 | About 2 to 3 hours the first time. The workflow starts the run and ends (GitHub's AWS login lasts one hour, so it cannot follow a run of this length). It stops at the approval and emails you. Run **pipeline** with `action = status` to see the step, the age and what is running on Fargate, and `action = stop` to end it |
| 3 | Review | open the MLflow URL and `s3://<bucket>/reports/<run>/` | metrics table, PR curves, figures |
| 4 | Promote | **promote** with the model version | Runs as the approver role; the GitHub user is recorded as reviewer |
| 5 | Approve | **approve** | Then the one frozen evaluation and the publish run |
| 6 | Serve | **deploy**, mode "full deploy" (or mode `auto`: it serves a newer approved model by itself) | Canary on Lambda. It deploys the newest model the pipeline published to S3 (`MODEL_VERSION` picks another); a real-data model without a recorded approval and frozen evaluation is refused |
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
RDS also rotates the database password every 7 days and the MLflow server reads it only when it starts, so stop and start the tools at least weekly (`power status` warns after 6 days). A Fargate Spot interruption of the tune or train step has no automatic retry: start the **pipeline** again with `prepare` off.

## Access and safety

- The tools have no login by design (course project). Their security groups accept nobody until `allow_cidr` opens one
  address for that start; **stop** closes it again. `0.0.0.0/0` needs the explicit `allow_anywhere` box.
- MLflow holds metrics, hashes and model files, and Evidently holds aggregates: no stay identifiers and no measurement
  values. The ward replay page from the **demo** workflow does hold PhysioNet trajectories, so it stays in the private bucket.
- Promotion runs under a different AWS role from training, and the GitHub environment's required reviewers (with
  "Prevent self-review") are the human approval. The next step refuses a model that was never promoted before it scores
  the test splits.

## How long a deploy takes, and why

| Part | Typical time | Why |
|---|---|---|
| Validate and preview | about 3 to 4 minutes (35 seconds before the ECS stacks existed; measured in runs 21 and 28) | CloudFormation prepares a change set for every nested stack |
| First `stack` run (new account or backend switch) | 15 to 30 minutes | RDS PostgreSQL takes 5 to 15 minutes to create; the Cloud Map namespace, cluster and roles are quicker; when a stack switches from SageMaker to ECS it also deletes the old pipeline stack and the SageMaker MLflow App at the end (cleanup) |
| `images` for a commit that never ran | first ever: 15 to 25 minutes; later commits: a few minutes | Every commit gets its own image tags. Dependencies are installed in a layer that only changes with a lock file, and each build reuses the previous image's layers, so a code-only commit rebuilds two small layers. The Fargate serving image is skipped in Lambda mode and the builds run side by side in the deploy workflow |
| Second `stack` run | 3 to 8 minutes | creates the MLflow and Evidently services, which wait for their task to start and pass its health check |
| A later deploy with no infrastructure change | about 8 to 12 minutes | validate, a no-op stack update, cached images |

If a service's task keeps crashing, the deployment circuit breaker marks the deployment failed after repeated failed launches, so the
update fails with a reason instead of waiting. The failure list shows only the latest attempt.

## If something fails

| Symptom | Look here |
|---|---|
| Pipeline says MLflow is stopped | **power** start |
| A step fails | the failure reason is in `s3://<bucket>/training-output/<run>/<step>/output/failure`; logs in `/ecs/<environment>-training` |
| `stack` cannot create the MLflow service | the tools image must exist first: run **deploy** again |
| Evidently report has no reference | **monitor** with action `reference` for the deployed model |
| Task cannot start | Fargate vCPU quota (**verify**; the default is 6 vCPU, the tools and a training task need 5), or lower `EcsTrainingCpu` |

What was checked against AWS documentation, and what still has to be watched on the first run, is in [aws-assumptions.md](aws-assumptions.md).

## Deploy: order, gates and recovery

The deploy workflow is a fixed sequence. Each step has a timeout, and a gate stops the run at the first step that fails, with the reason in the log.

| # | Step | What it guarantees | If it fails |
|---|------|--------------------|-------------|
| 1 | Validate and preview | Every template is valid and the change set builds. Nothing is created. | The message names the template or resource. Before the preview, an update left running by an earlier run is cancelled and waited for. |
| 2 | Check that a model is ready (full deploy only) | A published, approved model exists. Fails in seconds instead of after the image build. | Run the pipeline, then promote and approve. |
| 3 | Stack and images | stack (creates ECR repositories, no services yet) → images (built in parallel, cached) → stack (creates MLflow and Evidently with their image). A watcher prints why any ECS task stops and cancels the update if a service crash-loops. | The log shows the stopped reason and the last log lines of the container, then the stack failure events. |
| 4 | Wait for the hosted tools | MLflow and Evidently UI have all tasks running and healthy. A service switched off by the power workflow is skipped. | The same diagnosis, after 20 minutes at most. |
| 5 | Publish and serve (full deploy only) | The approved model is on Lambda behind the API. | See the publish log. |
| 6 | Smoke test, verify | `/health` and `/predict` answer. Every component exists and is healthy. | The verify table marks each failure. |
| 7 | Diagnose, next steps (always) | On failure: stack events, service states, stopped tasks with logs. A "what to do next" list goes into the run summary. | |

### Stack states and what the deploy does

A cancelled workflow does not stop CloudFormation, so a new run can find the stack mid-operation. Every state is handled before the preview:

| Stack state | The deploy |
|---|---|
| none, `CREATE_COMPLETE`, `UPDATE_COMPLETE`, `UPDATE_ROLLBACK_COMPLETE` | goes ahead (a rolled-back update is back to its last good state) |
| `UPDATE_IN_PROGRESS` | cancels that update, waits for the rollback, then goes ahead |
| `UPDATE_ROLLBACK_IN_PROGRESS`, `…CLEANUP_IN_PROGRESS` | waits |
| `CREATE_IN_PROGRESS`, `ROLLBACK_IN_PROGRESS`, `DELETE_IN_PROGRESS` | waits (a create cannot be cancelled) |
| `ROLLBACK_COMPLETE` (a first create failed) | deletes the empty stack, then creates it again |
| `UPDATE_ROLLBACK_FAILED` | stops. In the CloudFormation console choose **Continue update rollback**, then run deploy again. |
| `ROLLBACK_FAILED`, `DELETE_FAILED` | stops. Run the teardown workflow, which removes what blocks the delete, then deploy again. |

If the deploy job itself is cancelled while an update runs, it cancels the update before it exits, so nothing is left running unattended.

### Failure scenarios

| Symptom | Cause | What happens / what to do |
|---|---|---|
| Container stops at start (missing module, wrong setting) | image or settings wrong | The watcher prints the stopped reason and the log within about two minutes. After three stopped tasks with none running it cancels the update; deploy fails with that output. Fix, merge, run deploy. |
| Service never becomes healthy but does not crash | health check fails | The gate (step 4) times out and shows the stopped tasks and the service events. |
| `no approved published model` | full deploy before the pipeline | Step 2 stops the run. |
| Stack missing on a full deploy | first deploy in the account | Step 2 says to run **stack and images only** first. |
| Image already in ECR for this commit | re-run of the same commit | The image is reused, not rebuilt. |
| Two deploys at once | both started | The workflow's concurrency group queues the second; it never cancels a running deploy. |
| Billable services left on | demo finished | Run the **power** workflow with `stop`. |

Commands for a person with a shell: `bash scripts/aws_deploy.sh ready` (the gate), `bash scripts/aws_deploy.sh diagnose` (stack failures, then services and stopped tasks).
