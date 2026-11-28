# CI and CD: what runs, and when

Two rules:

* **Checks and builds run by themselves**, and only what the change needs.
* **Stack and model changes are started by hand.** The **deploy** workflow chooses its stages from what changed. The
  **images** workflow is the exception: it prebuilds and pushes commit-tagged images after qualifying pushes to `main`.
  Approval of a model stays a human step.

Every automatic workflow also has a **Run workflow** button, so you can start it yourself.

## Workflow organization

Workflow files stay directly in `.github/workflows/` because GitHub does not
load workflow files from subfolders. Their `name:` prefixes group them in the
Actions sidebar: `1 CI`, `2 Env`, `3 Model`, and `4 Ops`; filenames and run
history stay unchanged.

The local composites in `.github/actions/` hold the repeated setup:

- `aws-target` resolves the GitHub environment's region and role, then configures
  short-lived AWS credentials through OIDC.
- `python-tools` selects Python 3.12 and installs the selected hash-locked
  requirements file plus the local package.

If either shared action changes, run the workflow lint and focused helper tests.
`.github/CODEOWNERS` requests owner review for workflows, actions, CI rules,
infrastructure, AWS scripts, and dependency locks. To make that review
mandatory, enable **Require review from Code Owners** for `main`.

## Automatic

| When | Workflow | Does | AWS access |
|---|---|---|---|
| Pull request, push to main | **tests** | Classifies the change, then runs only the jobs it needs (table below). `ci-ok` is the one required check. | none |
| Weekly | **tests** | Known-vulnerability scan of the four lock files | none |
| Push to main that touches `Dockerfile`, `requirements/`, `src/` or `pyproject.toml` | **images** | Builds the images for that commit and pushes them to ECR, so deploy skips the build | writes images only |
| Weekly | **dependabot** | Pull requests that move the pinned GitHub Actions forward | none |

### What the tests workflow runs

`scripts/ci_changes.py` sorts every changed file into a category with `ci/change_rules.json` (first matching pattern wins), and each
category names its CI jobs and deploy stages. A file nobody listed counts as `other` and runs the unit tests.

| You changed | CI jobs | A deploy would run |
|---|---|---|
| docs, README, reports | only the patient-data guard | nothing |
| tests, notebooks | unit | nothing |
| `.github/` | workflow lint (actionlint) | nothing |
| serving code (API, scoring, alerts, model loading) | unit, docker | serve |
| training code (`src/`, tuning, prepare, models) | unit, docker | retrain |
| features, schema, policy | unit, docker | retrain and serve |
| MLflow or Evidently code | unit, docker | stack |
| `infra/`, `scripts/aws_*` | unit, template checks | stack |
| `Dockerfile` | unit, docker | stack and serve |
| `requirements/`, `pyproject.toml` | unit, docker, vulnerability scan | stack and serve |

`ci-ok` passes when every job that was needed passed. A job that was not needed is skipped, which is fine. Make `ci-ok` the
required status check in the branch protection rule. Manual run: tick `run_everything` to run all jobs, or untick it to compare
with main.

## Manual: deploy

**deploy** → Run workflow. Inputs:

| Input | Default | Meaning |
|---|---|---|
| `environment`, `region` | dev | where |
| `mode` | `auto` | `auto` runs only the stages the changes need. `stack and images only` and `full deploy` force a fixed set |
| `retrain` | `auto` | `auto` starts a real-data training run only when training code changed, or none was ever run. `yes` or `no` overrides |
| `force_prepare` | off | with a retrain: prepare the data again. The skip compares only the source, limit and seed, so change the preparation code, then tick this |
| `dry_run` | off | print the plan and run nothing |
| `model_version`, `model_uri` | empty | which published model to serve (empty = newest approved) |

### How the plan is made

After each successful deploy the workflow writes `deployments/<env>/state.json` to the artifact bucket: the commit each stage last
ran for (`stack`, `retrain`, `serve`) and the model version last served. The next deploy compares this commit with each stage's
commit, files changed since then go through the same rules as CI, and the facts of the account adjust the result:

* **Serve** needs an approved, published model. It is also due when a newer model than the served one exists.
* **Retrain** is skipped while a pipeline run is in progress, and when a model already exists in an environment that never trained.
* Nothing but the stack can run before the stack exists.
* `mode` and `retrain` override all of that. A full deploy that was asked for fails at once, with what to do, when no model is published.

The plan is printed in the run summary with the reason for each stage. A run that finds nothing to do says so and ends.

### Stages, in order

1. **Plan**
2. **Validate**: template check and change-set preview, nothing created
3. **Images** (reused if the images workflow built them) and **stack**: update, then wait for MLflow and Evidently to be healthy
4. **Retrain**: copy the real data once (a marker), switch MLflow on if it is off, start the pipeline, do not wait for it
5. **Serve**: publish the approved model, update the Lambda, smoke test
6. **Verify**: every component exists and is healthy
7. **Remember**: write the state file. Only when everything passed

### The usual flow on a new account

| Click | What the plan does |
|---|---|
| 1 | Nothing exists: creates the stack and images. Says training comes next |
| 2 | Stack is up, nothing trained: starts the real-data run, which ends at the approval email |
| (you) | **promote**, then **approve** |
| 3 | A model is published and none is served: serves it, runs the smoke test |
| later | Only what changed: docs do nothing; serving code serves again; training code retrains |

## What stays manual, and why

* **promote** and **approve**: a different person reviews and approves each model. This is the governance control; the workflow
  that starts training cannot approve its own model.
* **power**, **demo**, **monitor**, **performance**, **pipeline**, **teardown**, **verify**: they act on running resources or cost money. They work as before.

## Good to know

* Any stack update moves the MLflow and Evidently services to the new commit's image. It is a rolling replace and takes a minute or two.
* A retrain records the commit when it **starts**. If the run fails, start it again with `retrain` = yes.
* The images workflow skips quietly when there is no stack, and can be turned off with the repository variable `AUTO_PREBUILD` = `false`.
* Pull requests from forks never get AWS access: CI runs without any secrets.
* AWS workflows serialize by environment, including runs targeting different regions. This prevents a region alias such as
  `environment default` from bypassing the deployment lock.
