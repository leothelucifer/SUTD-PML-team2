# Team Task Tracker

## How to update

- Each owner updates their task rows as work progresses.
- Use **Not started**, **In progress**, **Blocked**, **In review**, **Done**, or **Deferred** for status.
- Set a due date and last-updated date using `YYYY-MM-DD`. A dash means not set.
- Add a PR, test result, or document link in **Evidence / notes**. For blocked tasks, name the blocker and next action.
- Mark a task **Done** only after its deliverable is verified and reviewed. Link private data evidence without uploading patient data.
- Statuses start as **Not started** until owners confirm existing work; they are not an assessment of current implementation.
- Share updates through commits or PRs once this project is in a shared Git repository. Markdown tables are manually maintained.

## Owners and slides

| Person | Owner | Slides |
|---|---|---|
| 1 | Edwin | 4, 5, 8 |
| 2 | Jawad | 6 |
| 3 | Kaiyuan | 3, 7, 9 |
| 4 | Josh | 11 |
| 5 | leo | 13, 14 |
| 6 | Mashie | 10, 12, 15, 16 |

## Task status

### Person 1 - Edwin

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| E01 | Download data and convert to Parquet | Not started | - | - | - |
| E02 | Version data with DVC | Not started | - | - | - |
| E03 | Data schema checks with Pandera | Not started | - | - | - |
| E04 | Hospital A / B split | Not started | - | - | - |
| E05 | Exploratory analysis | Not started | - | - | - |
| E06 | SIRS rule baseline | Not started | - | - | - |
| E07 | Prepare slides 4, 5, 8 | Not started | - | - | - |

### Person 2 - Jawad

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| J01 | Causal rolling features | Not started | - | - | - |
| J02 | Missing-value handling | Not started | - | - | - |
| J03 | Hours since last lab test | Not started | - | - | - |
| J04 | Leakage tests: corrupt hour t+1 and verify earlier features remain unchanged | Not started | - | - | - |
| J05 | Fast utility-score function | Not started | - | - | - |
| J06 | Shared feature module | Not started | - | - | - |
| J07 | Prepare slide 6 | Not started | - | - | - |

### Person 3 - Kaiyuan

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| K01 | Logistic regression | Not started | - | - | - |
| K02 | Random forest | Not started | - | - | - |
| K03 | LightGBM as the main model | Not started | - | - | - |
| K04 | Probability calibration | Not started | - | - | - |
| K05 | Alarm threshold and six-hour cooldown | Not started | - | - | - |
| K06 | Fairness checks | Not started | - | - | - |
| K07 | Prepare slides 3, 7, 9 | Not started | - | - | - |

### Person 4 - Josh

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| H01 | HMM forward-pass states | Not started | - | - | - |
| H02 | Signature features | Not started | - | - | - |
| H03 | With / without tests | Not started | - | - | - |
| H04 | Optional: GMM | Not started | - | - | - |
| H05 | Optional: GRU | Not started | - | - | - |
| H06 | Prepare slide 11 | Not started | - | - | - |

### Person 5 - leo

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| L01 | MLflow tracking and registry | Not started | - | - | - |
| L02 | FastAPI service | Not started | - | - | - |
| L03 | Docker image | Not started | - | - | - |
| L04 | CI/CD with GitHub Actions | Not started | - | - | - |
| L05 | Latency tests | Not started | - | - | - |
| L06 | AWS deployment design | Not started | - | - | - |
| L07 | Prepare slides 13, 14 | Not started | - | - | - |

### Person 6 - Mashie

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| M01 | Hospital B replay | Not started | - | - | - |
| M02 | Drift monitoring at two speeds | Not started | - | - | - |
| M03 | SHAP explanations | Not started | - | - | - |
| M04 | Model card | Not started | - | - | - |
| M05 | Risk register | Not started | - | - | - |
| M06 | Final deck assembly | Not started | - | - | - |
| M07 | Prepare slides 10, 12, 15, 16 | Not started | - | - | - |

## Shared project boundaries

- Keep patient data under the Git-ignored `data/` directory. DVC metadata may be versioned; patient data must not be committed to Git.
- Preserve record-level split isolation. Use training data for fitting, calibration data for calibration, and validation data for model and threshold selection.
- Freeze model, feature, calibration, and alarm-policy decisions before Hospital B replay or held-out evaluation.
- Use only current and past observations for features; do not shift the supplied benchmark label again.
- Treat these assignments as planned work, not confirmation that the listed tools or capabilities are already implemented.
