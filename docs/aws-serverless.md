# Serverless AWS: SageMaker training, SageMaker MLflow and Lambda scoring

The default AWS design runs only when it is used. Nothing in the default stack
bills by the hour: no load balancer, NAT gateway, VPC endpoint, notebook,
tracking server or idle container.

| Step | Service | How it is serverless |
|---|---|---|
| Experiment tracking and model registry | **SageMaker AI MLflow App** (`AWS::SageMaker::MlflowApp`) | Serverless MLflow at no additional charge; you pay for its S3 storage. Replaces the hourly MLflow tracking server |
| Training and selection | **SageMaker AI training job** started by `scripts/sagemaker_train.py` | One instance for the length of the job, on **managed spot** capacity by default, with a hard maximum runtime |
| Scoring API | **AWS Lambda** container image with a **function URL** | Scales to zero; billed per request and millisecond; reserved concurrency caps the cost |
| Storage | S3 (KMS), ECR, DynamoDB on demand | Pay per GB or per request; lifecycle rules expire old versions, images and training outputs |
| Alerts | CloudWatch alarms, SNS email, AWS Budget | Budget alert exists from the first deploy |

The always-on private design from the deck (internal load balancer and Fargate
in private subnets behind a hospital VPN) is still available as
`ServingMode=fargate`.

**Status:** written and statically checked (cfn-lint, botocore request
validation, unit tests, and local container runs of the Lambda and training
images). Not yet deployed to an AWS account. Deploying needs the L06 owner's
approval of the account, region, budget and data plan, and real patient data
must not be uploaded until licence, privacy and residency are confirmed.

## How it fits together

```text
laptop / CI                              AWS account (ap-southeast-1)
-----------                              ----------------------------
docker build --target training  ──push──▶ ECR  <env>/training
docker build --target lambda    ──push──▶ ECR  <env>/scoring-lambda
data/physionet/{combined.parquet,         S3   <bucket>/data/physionet/   (KMS)
  splits.csv, summary}          ──copy──▶
scripts/sagemaker_train.py      ──start─▶ SageMaker training job (spot, 1 instance, max 1 h)
                                            reads data/, runs `train`, writes training-output/
                                            logs to + registers in ──────▶ MLflow App  (@candidate)
promote --reviewed-by           ────────────────────────────────────────▶ MLflow App  (@production)
publish                         ──copy──▶ S3   <bucket>/models/vN/model.joblib (+ validation summary)
stack update (ModelS3Uri, ModelSha256) ─▶ Lambda function (image + pinned model)
hospital system / call_scoring_url.py ── SigV4-signed HTTPS ──▶ function URL ─▶ /health /model /score /predict
```

The Lambda function and the Fargate task load the bundle from S3 and refuse it
unless its SHA-256 matches the pinned `ModelSha256`, so a changed object is
never loaded.

## Commands

Prerequisites: AWS CLI v2 with credentials for the account, Docker, and
`pip install -e ".[aws,tracking,sagemaker]"`. Attach the stack's
`TeamOperatorPolicyArn` output to the people (or CI role) who run these steps.

```bash
export AWS_REGION=ap-southeast-1 STACK=sepsis-research
export BUDGET_USD=20 BUDGET_EMAIL=you@example.com        # monthly budget alert

bash scripts/aws_deploy.sh stack          # S3, KMS, ECR, MLflow App, roles, alarms, budget
bash scripts/aws_deploy.sh images         # build + push lambda, training and scoring images (tag = git sha)
bash scripts/aws_deploy.sh data           # upload combined.parquet, splits.csv and its summary (after approval)
bash scripts/aws_deploy.sh train          # spot training job; registers the selected model as @candidate
bash scripts/aws_deploy.sh mlflow-ui      # short-lived link to the MLflow UI: compare runs, review the candidate
bash scripts/aws_deploy.sh promote 1 "Reviewer Name"
bash scripts/aws_deploy.sh publish        # copies @production to S3, records the hash
bash scripts/aws_deploy.sh serve          # deploys the Lambda scoring API with that model
bash scripts/aws_deploy.sh call GET /model
bash scripts/aws_deploy.sh call POST /predict \
  '{"stays": [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 92}, {"ICULOS": 2, "HR": 118}]}]}'
```

To release a new model: `train`, review in the MLflow UI, `promote`, `publish`,
`serve`. To roll back: `promote` the previous version, then `publish` and
`serve` again. `train --on-demand` skips spot capacity when a result is needed
quickly. `SERVING_MODE=fargate` deploys the always-on private design instead.

The same commands work without AWS for local runs: `train`/`demo` with
`--mlflow-uri sqlite:///artifacts/mlflow.db`, and the MLflow App ARN in place
of the SQLite URI once the stack exists.

## Rough monthly cost

Estimates for `ap-southeast-1`, light research use (one ward polled hourly, a
few training runs a month). Check them in the AWS Pricing Calculator before
deploying; prices change and these are not quotes.

| Item | Serverless default | Notes |
|---|---|---|
| Lambda scoring | about $0 | ~720 requests and ~720 GB-seconds a month fit in Lambda's always-free monthly allowance |
| MLflow App | $0 + S3 storage | No hourly charge (a Small tracking server would cost roughly $430 a month if left on) |
| Training jobs | cents per run | `ml.m5.xlarge` costs about $0.25-0.30 an hour on demand; spot is usually much less, and a run takes minutes |
| S3, ECR | about $1-2 | A few GB of data, models and images; old versions and images expire |
| KMS key | $1 | One customer-managed key |
| CloudWatch alarms and logs | under $1 | 3 alarms; logs kept 30 days |
| **Total** | **roughly $2-5 a month** | plus training runs |
| Fargate mode instead | roughly $90-100 a month before traffic | Load balancer, three interface VPC endpoints in two zones, one task running all the time |

Costs that remain even when idle: the KMS key, stored data and images, and
alarms. `aws_deploy.sh down` deletes the stack; the bucket, key, repositories,
table and log groups are retained on purpose and must be emptied and deleted by
hand when the project ends.

## Security notes

- **Authentication.** The function URL uses `AWS_IAM`: every request must be
  SigV4-signed by a principal allowed to call it (`scripts/call_scoring_url.py`
  shows how). An outside principal, such as a hospital integration role, is
  allowed with `AllowedInvokerArn`. The deck's OAuth2 API Gateway and VPN are
  not built; the Fargate mode keeps the network-private option.
- **Network.** In Lambda mode the function is not in a VPC. Requests reach it
  over HTTPS on AWS's public endpoint and are rejected without valid
  signatures; the function reaches S3 with its IAM role.
- **Data.** The artifact bucket is private, versioned, KMS-encrypted and denies
  non-TLS access. Training outputs (`model.tar.gz`) contain the full run,
  including patient-level held-out predictions, so they stay in this bucket
  and expire after `TrainingOutputRetentionDays`. Nothing patient-level is
  logged to MLflow, and `/model` serves validation results only.
- **Least privilege.** The training role reads `data/`, writes
  `training-output/` and calls only this MLflow App. The Lambda role reads only
  `models/`. The MLflow App's role reaches only `mlflow/`.
- **Unverified in an account.** The IAM action for logging to an MLflow App is
  `sagemaker:CallMlflowAppApi` (the sagemaker-mlflow plugin signs App requests
  for the `sagemaker` service). Check it on the first real training run.
