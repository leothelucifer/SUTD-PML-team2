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
| E01 | Download data and convert to Parquet | Done | - | 2026-11-09 | Full data: 40,336 records / 1,552,210 rows; 0 SHA-256 mismatches vs manifest. `tests/test_prepare.py` (corrupt input, overwrite refusal, values kept exactly). Merged by Leo in PR #48; no formal Approve review. |
| E02 | Version data with DVC | Not started | - | 2026-11-09 | Proposed for dropping: no shared storage, everyone downloads the data. Awaiting team reply. |
| E03 | Data schema checks with Pandera | Not started | - | 2026-11-09 | Proposed for dropping: not required by the brief; `prepare` already rejects bad data. Awaiting team reply. |
| E04 | Hospital A / B split | Done | - | 2026-11-09 | `split` command writes `splits.csv` (seed 42, hashes). 55/15/15/15 of A, all B external, every stay once, load-order independent. `tests/test_splits.py`. Stay-level, not person-level. Merged by Leo in PR #48; no formal Approve review. `baseline` can now read the saved split. |
| E05 | Exploratory analysis | In progress | - | 2026-11-09 | `compare-hospitals` command, 7 figures in `reports/figures/`, findings in `docs/hospital-a-vs-b.md`. A restricted to train split (no test data). To do: team OK on descriptive use of B. Hospital B wording on hold until the model owner replies. |
| E06 | SIRS rule baseline | Done | - | 2026-11-09 | `baseline` command → `baseline_metrics.csv`; SIRS F1 0.063 (A val), utility −0.014 at B. Independently recomputed with the official utility formula (match to 1e-15). Adds accuracy, specificity, confusion matrix and a validation-only threshold sweep (fig 8). `tests/test_baselines.py`. Merged by Leo in PR #48; no formal Approve review. Failure patterns: `docs/sirs-failure-patterns.md`. |
| E07 | Prepare slides 4, 5, 8 | Not started | - | 2026-11-09 | Deck README now lists slides 4, 5, 9, 10; awaiting confirmation. Slides done last. |

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
