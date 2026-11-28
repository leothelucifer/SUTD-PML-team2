# Training on ECS Fargate (no SageMaker quota needed)

A new AWS account has a SageMaker training-instance quota of 0, and a free-plan account may not be able
to raise it. `TrainingBackend=ecs` runs the same pipeline steps (prepare, tune, train, the approval, the
frozen evaluation and the publish) as **ECS Fargate tasks** instead of SageMaker training jobs. Serving,
MLflow, promotion, approval and monitoring are unchanged.

| | `sagemaker` (default) | `ecs` |
|---|---|---|
| Where a step runs | SageMaker AI training job | ECS Fargate task (same training image) |
| Quota that must be above 0 | SageMaker training instance (on demand and spot) | Fargate vCPU (on demand, and spot if used) |
| Entry point | `sepsis_warning.adapters.aws.sagemaker_train` | `sepsis_warning.adapters.aws.ecs_job`, which stages the S3 channels and then calls the same entry point |
| Pipeline template | `pipeline.yaml` | `pipeline-ecs.yaml` (same state names) |
| Extra infrastructure | none | `ecs-platform.yaml`: a small VPC with public subnets, a cluster, task roles, a log group |
| Cost while idle | none | none (no NAT gateway, endpoints, load balancer or database) |

## Turn it on

`infra/params/dev.json` already sets `TrainingBackend` to `ecs`. For another environment, set the
parameter in its params file, or for one run `TRAINING_BACKEND=ecs bash scripts/aws_deploy.sh stack`.
Then run the **deploy** workflow (mode `auto`, or "stack and images only") and the **pipeline** workflow as usual. The
pipeline workflow's `instance_type` input is ignored by this backend; the task size comes from
`EcsTrainingCpu` and `EcsTrainingMemory` (default 4 vCPU, 16 GB).

## Check the Fargate quota first

The **verify** workflow reports it. By hand, in the same region as the stack:

```bash
aws service-quotas list-service-quotas --service-code fargate --region ap-southeast-2 \
  --query "Quotas[?contains(QuotaName,'vCPU')].[QuotaName,QuotaCode,Value]" --output table
```

The default quota in a new account is **6 vCPU per Region for On-Demand and 6 for Spot** ([ECS quotas](https://docs.aws.amazon.com/general/latest/gr/ecs-service.html)). A training task needs the vCPU you set in `EcsTrainingCpu` (4 by default), and the MLflow server (0.5) and the Evidently UI (0.5) run at the same time, so 5 are needed. Memory must be a valid pair for the CPU value ([task sizes](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html)); the deploy refuses an invalid pair. If the value is lower, either
request an increase in Service Quotas or choose a smaller task (`EcsTrainingCpu=2048`, `EcsTrainingMemory=8192`).
Fargate Spot has its own quota: set `EcsUseSpot=false` to use on-demand capacity only. `EcsUseSpot` is `true` in every parameter file, so the tune and train steps (the long ones) already run on Spot; the short prepare and evaluate steps and the MLflow and Evidently services stay on demand.

## What a run costs (estimate)

Fargate bills per second for the vCPU and memory of a running task. Rough Sydney prices are about
$0.05 per vCPU-hour and $0.006 per GB-hour; Fargate Spot is much cheaper. Check the AWS pricing page for
current values. A 4 vCPU / 16 GB task costs about $0.30 an hour on demand. A real-data run (prepare about
15 minutes, tune and train about 2 hours, publish 10 minutes) is therefore under $1, and a synthetic
rehearsal a few cents. Public IPv4 addresses on the tasks cost about half a cent per hour each.
Nothing bills between runs. Setting `MonthlyBudgetUsd` keeps the account's budget alert in place.

## Where outputs go

Each step writes to `s3://<bucket>/training-output/<execution name>/<step>/`: `model/` (what SageMaker would
have archived) and, when a step fails, `output/failure` with the reason. The execution name is the pipeline
run's name. Task logs are in the CloudWatch log group `/ecs/<environment>-training`.

## Security notes

- Tasks accept no inbound connections (no ingress rules) and can only open HTTPS. They get a public IP
  because the VPC has no NAT gateway; that is deliberate, to avoid a standing cost.
- The task role has the same data permissions as the SageMaker training role (read `data`, `tuning`, `raw`;
  write `data`, `tuning`, `reports`, `training-output`, `models`; use the artifact key; call the MLflow App).
  The execution role can only pull the image and write logs.
- The pipeline's role may run tasks only on this cluster and pass only these two roles.
- Patient-level test predictions from the frozen evaluation are not written to the step's output.
