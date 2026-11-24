# AWS assumptions and where each was checked

Everything in the ECS, RDS and Step Functions work that depends on how AWS behaves was checked against AWS's own documentation
(read on 2026-11-26 through the sandbox proxy) or against AWS's API models. Two kinds of check are automated in
`tests/test_aws_api_shapes.py`: every parameter name in the state machines and every call the scripts make must exist in the
real API definitions (the service models shipped in botocore), and every field the code reads from a response must exist in
the real response shape. The IAM actions in the templates were looked up in the Service Authorization Reference
(47 actions, none unknown). Behaviour that no automated check can see is listed here with its source.

## Confirmed in the documentation

| Assumption | Result | Source |
|---|---|---|
| Step Functions parameters for `ecs:runTask.sync` are PascalCase (`Cluster`, `TaskDefinition`, `Overrides.ContainerOverrides[].Environment`) | Confirmed: "Parameters in Step Functions are expressed in PascalCase", with `Overrides.ContainerOverrides[].{Name,Environment}` in the example | [Run ECS or Fargate tasks with Step Functions](https://docs.aws.amazon.com/step-functions/latest/dg/connect-ecs.html) |
| `.sync` needs `ecs:RunTask`, `ecs:StopTask`, `ecs:DescribeTasks`, `events:PutTargets/PutRule/DescribeRule` on rule `StepFunctionsGetEventsForECSTaskRule`, and `iam:PassRole` for the execution and task roles | Confirmed (same page, IAM section; the page's own policy uses `Resource: "*"` for Stop/Describe, ours is narrower and valid per the reference below) | same |
| A non-empty `Failures` field fails a `.sync` task with `AmazonECS.Unknown` | Confirmed | same |
| AWS SDK integration resource form and casing: `arn:aws:states:::aws-sdk:ecs:registerTaskDefinition`, PascalCase parameters | Confirmed | [AWS SDK integrations](https://docs.aws.amazon.com/step-functions/latest/dg/supported-services-awssdk.html) |
| SNS `MessageAttributes` take `DataType` and `StringValue` | Confirmed (shape also checked against the SNS API model) | [Publish to SNS with Step Functions](https://docs.aws.amazon.com/step-functions/latest/dg/connect-sns.html) |
| `States.Format` needs one `{}` per argument; `'`, `{`, `}` and `\\` in the template must be escaped | Confirmed; no template string contains an apostrophe or a backslash, and a test counts placeholders | [Intrinsic functions](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-intrinsic-functions.html) |
| History event `TaskScheduled` carries the parameters as a JSON string | Confirmed: `parameters` is "the JSON data passed to the resource referenced in a task state" (String); `resource` is "the action of the resource called" | [TaskScheduledEventDetails](https://docs.aws.amazon.com/step-functions/latest/apireference/API_TaskScheduledEventDetails.html) |
| ECS Fargate task sizes: 512 CPU with 2048 MiB, 4096 CPU with 16384 MiB, 512 CPU with 1024 MiB, 1024 CPU with 4096 MiB | Confirmed from the table; `root.yaml` now has one Rule per CPU value so an invalid pair stops the deploy | [Task definition parameters](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html) |
| Secret injection format `...:secret:name:json-key:version-stage:version-id` with the execution role allowed to read the secret | Confirmed (we use `<arn>:password::`) | [Secrets Manager secrets in ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/secrets-envvar-secrets-manager.html) |
| Service discovery with awsvpc uses A records; the VPC needs DNS support and hostnames; Fargate platform 1.1.0 or later; records hold the **private** address | Confirmed | [ECS service discovery](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-discovery.html) |
| `aws:SourceAccount` and `aws:SourceArn` are the recommended conditions in the task role trust policy (SourceArn cannot name one cluster) | Confirmed (we use `aws:SourceAccount`) | [Task IAM role](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html) |
| ECS tag keys may contain `:` (`power:group`) and may not start with `aws:` | Confirmed | [Tagging ECS resources](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-using-tags.html) |
| `RunTask` takes `ecs:cluster`; `DescribeTasks` and `StopTask` have resource type `task` (`task/${ClusterName}/${TaskId}`) | Confirmed | [Service Authorization Reference for ECS](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonelasticcontainerservice.html) |
| Fargate quotas are named "Fargate On-Demand vCPU resource count" and "Fargate Spot vCPU resource count", **default 6 vCPU each, per Region** | Confirmed; verify now asks for 5 (training 4 + MLflow 0.5 + Evidently UI 0.5) | [ECS endpoints and quotas](https://docs.aws.amazon.com/general/latest/gr/ecs-service.html) |
| Fargate Spot tasks are interrupted with a two-minute warning; the capacity provider must be associated with the cluster first | Confirmed (the cluster lists FARGATE and FARGATE_SPOT) | [Fargate capacity providers](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-capacity-providers.html) |
| RDS managed master password: stored in Secrets Manager and **rotated every seven days by default** | Confirmed. Consequence: the MLflow server reads the password only when it starts, so `power status` warns after 6 days and the runbook says to stop and start it weekly | [Password management with Secrets Manager](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/rds-secrets-manager.html) |
| A stopped RDS instance is not billed for instance hours, still for storage and backups, and **restarts after seven days** | Confirmed; a single-AZ PostgreSQL instance without replicas can be stopped | [Stopping an RDS instance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_StopInstance.html) |
| PostgreSQL 17.9 and 17.11 exist and are current; `db.t4g.micro` exists | Confirmed; the template now pins **17.11** (standard support to September 2027) instead of 17.9 (to March 2027) | [RDS for PostgreSQL release calendar](https://docs.aws.amazon.com/AmazonRDS/latest/PostgreSQLReleaseNotes/postgresql-release-calendar.html) |
| Tasks whose security group allows only HTTPS (443) and port 5000 out can still resolve `mlflow.<env>.local` and AWS names | Confirmed: "Security groups cannot block DNS requests to or from the Route 53 Resolver" | [DNS attributes in your VPC](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-dns.html) |
| `cloudwatch:PutMetricData` can be limited with `StringEquals` on `cloudwatch:namespace`, resource `*` | Confirmed | [Condition keys for CloudWatch namespaces](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/iam-cw-condition-keys-namespace.html) |
| Alarm `Period` may be any multiple of 60 (we use 86400) | Confirmed | [PutMetricAlarm](https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_PutMetricAlarm.html) |

## Not stated in the documentation (handled so a wrong guess cannot break a run)

| Question | What the documentation says | How the pipeline copes |
|---|---|---|
| Does `ecs:runTask.sync` fail the state when a container exits non-zero? | The Step Functions page does not say | A failure raised by the integration goes to `Catch`; a finished task whose result has `Containers[0].ExitCode` that is not 0 goes to the failure notice; a result without that field continues, and later steps refuse bad inputs (an unpromoted model is refused before the test split is scored). **Check the first execution's output.** |
| The exact output keys of `runTask.sync` | Not stated | Nothing reads them except the optional exit-code check above; there is no `ResultSelector` on the task states |
| The exact value of `taskScheduledEventDetails.resource` for the approval state | Described as "the action of the resource", no example | The approval script matches any resource containing `waitForTaskToken` and skips events whose parameters are not JSON |
| Whether a Spot interruption of the tune or train step can be retried safely | Interrupted tasks get a two-minute warning; nothing about Step Functions retries | No automatic retry (a failing gate would be retried too). Start the pipeline again with `prepare=false` |

## Found wrong or risky by this check, and changed

- The runTask exit-code check used a `ResultSelector` on an output shape the documentation does not define: removed.
- The default Fargate quota is 6 vCPU, not enough for a second 4 vCPU task next to the tools: verify and the runbook say so.
- The RDS password rotates every 7 days while the MLflow server reads it once: warning in `power status`, runbook note.
- RDS PostgreSQL 17.9 reaches the end of standard support sooner than 17.11: pinned to 17.11.
- The memory default (16384) is invalid for 1024 or 2048 CPU units: rule added in `root.yaml`.

## Not yet confirmed by running it

None of this has run in an AWS account yet. The first execution should be watched for: the shape and exit-code behaviour of
`runTask.sync` (above), MLflow starting against RDS, the Cloud Map name resolving from a training task, and the first
`power start` with an `allow_cidr`.
