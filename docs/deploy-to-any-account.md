# Deploy to any AWS account

Three steps, run once per account. They work the same for dev, staging and prod,
in any region. Nothing in the repository is specific to one account.

## Before you start

- An AWS account and a shell with admin credentials for it (for example
  CloudShell, or a profile in `~/.aws`). This is used only for the one-time setup.
- AWS CLI v2 and Docker (for the image build).
- The GitHub CLI (`gh`), logged in, if you want the variables set automatically.
  Otherwise the script prints the commands to run.
- The region you want. The SageMaker MLflow App must be available in it; check
  the list in `docs/aws-serverless.md`. Both Singapore (`ap-southeast-1`) and Sydney (`ap-southeast-2`) are supported and checked; the examples use Sydney, so replace it with `ap-southeast-1` for Singapore.

## 1. Connect GitHub to the account (once per account and region)

```bash
export AWS_REGION=ap-southeast-2
export GITHUB_REPO=owner/SUTD-PML-team2       # the repository that will run the deploy workflow
bash scripts/bootstrap_github_oidc.sh
```

This creates the GitHub OIDC identity provider (only if the account has none),
and one deploy role each for `dev`, `staging` and `prod`. It sets the GitHub
variables `AWS_REGION` and `AWS_DEPLOY_ROLE_ARN` for each environment. No access
keys are created.

Then, in GitHub, under Settings → Environments, add required reviewers to
`staging` and `prod`.

The deploy roles start with `AdministratorAccess`, because the first deployment
creates IAM roles. After the first successful deploy, replace it with a narrower
policy: pass `DEPLOY_POLICY_ARN=<arn>` to the script, or edit the role in IAM.

## 2. Create the stack and the first model (from the repository)

Option A, from GitHub (recommended):

1. Actions → **deploy** → Run workflow, environment `dev`.
   The first run fails at `publish` if no model has been promoted in this
   account yet. That is expected: step 3 below creates the model first.

Option B, from your shell (for a first look, or without GitHub):

```bash
export AWS_REGION=ap-southeast-2 ENVIRONMENT=dev BUDGET_EMAIL=you@example.com
bash scripts/aws_deploy.sh stack          # S3, KMS, ECR, MLflow App, roles, pipeline, alarms
bash scripts/aws_deploy.sh images         # builds and pushes the three images
```

## 3. Train, review and promote a model

The training pipeline is the production route: it trains, checks the promotion
gates, waits for a named approver, and publishes. Its first use in an account:

```bash
bash scripts/aws_deploy.sh data           # after data-use approval; synthetic is fine for a dev rehearsal
bash scripts/aws_deploy.sh pipeline       # starts training and waits for approval
bash scripts/aws_deploy.sh approve <token> # the token arrives by email; the approver is not the trainer
```

For a dev rehearsal on synthetic data, set `ALLOW_SYNTHETIC=1`. Then run the
deploy workflow again (or `bash scripts/aws_deploy.sh publish && serve`) to put
the model behind the scoring API.

## Checking it worked

```bash
bash scripts/aws_deploy.sh call GET /health     # the bundle hash should match the published model
bash scripts/aws_deploy.sh outputs              # every stack output, including the API URL
```

## Moving to another account

Repeat step 1 in the new account, with its region. Then run the deploy workflow
with `ENVIRONMENT` set to that account's environment. The code and the templates
do not change between accounts.

## What the setup creates

| Resource | Why |
|---|---|
| GitHub OIDC provider (IAM) | Lets GitHub Actions obtain short-lived credentials without stored keys |
| Three deploy roles | One per GitHub environment; the trust policy accepts only jobs in that environment |
| Stack `sepsis-github-oidc` | Holds the above; delete it to remove the connection |

## Troubleshooting

- **"missing AWS_REGION or AWS_DEPLOY_ROLE_ARN"** in the workflow: run step 1
  for that account, or set the variables for that GitHub environment.
- **"set AWS_REGION"** from a script: export `AWS_REGION` in the shell first.
- **"this environment sets a monthly budget; set BUDGET_EMAIL"**: set `BUDGET_EMAIL`,
  or set `MonthlyBudgetUsd` to 0 in `infra/params/<env>.json`.
- **Role trust errors (`Not authorized to perform sts:AssumeRoleWithWebIdentity`)**:
  the repository name in `GITHUB_REPO` must match the repository running the workflow,
  and the job must run in the matching GitHub environment.
- **MLflow App not found in the region**: choose a region from the list in
  `docs/aws-serverless.md`.
