# Clinical-use design and validation plan

**Status:** Proposed research design; not approved for clinical use

**Date:** 2026-11-07

This document translates the project's early-warning research question into a
candidate clinical use and an evidence plan. The proposed use and thresholds
must be reviewed by clinicians, hospital operations, and the appropriate
research and regulatory governance bodies before any clinical-facing use.

## 1. Intended purpose

### Candidate user and setting

- **Primary user:** a bedside clinician responsible for a patient in an ICU.
- **Setting:** an ICU workflow using the hourly measurements represented in
  the source data. The current evidence does not support generalization to
  emergency departments, wards, outpatient care, or all hospital patients.
- **Candidate purpose:** identify an ICU stay that may be approaching a
  prospectively defined sepsis event, early enough to prompt clinical review.

This is a proposal for a research target, not a statement that the current
model can identify sepsis or improve care.

### Candidate action

If a future system produces a validated alert, its only proposed action is to
ask the responsible clinician to review the patient's current condition and
follow the institution's existing assessment process. It must not diagnose
sepsis, recommend or initiate treatment, or replace clinician judgment.
Whether even this prompt is useful and safe must be tested with intended users.

### Candidate prediction target

For an eligible patient-hour before the event, estimate the risk of a
**first, clinically defined sepsis onset within the next six hours**. The
event definition, time of onset, eligible population, and handling of discharge,
death, and incomplete observation must be specified with clinical partners
before training or evaluating a model for this target.

The current `SepsisLabel` is not that target: the challenge label is shifted
six hours before its dataset-defined onset and remains positive afterward.
The existing model therefore estimates the challenge-label probability. It
must not be presented as a validated six-hour clinical risk. A label-construction
and audit step is required before claiming that the proposed target can be
derived reliably from these records.

## 2. What success means

The primary evaluation should be event- and workflow-oriented, not only
hour-row classification:

1. **Timely event detection:** proportion of eligible events with at least one
   alert in a pre-specified, clinically meaningful interval before onset.
2. **Alert burden:** false alert episodes per observed patient-day, using
   explicit denominators and counting cooldown-suppressed notifications
   separately from hourly model-positive rows.
3. **Lead time:** distribution of time from the first qualifying alert to the
   event, including censored and unobservable cases.
4. **Reliability and equity:** calibration, missing-data behavior, and
   event-detection and alert-burden results across clinically relevant groups
   and sites, with sample sizes and uncertainty.
5. **Workflow value:** in a later prospective study, whether clinicians
   receive, understand, and can respond to alerts without unacceptable
   interruption, delay, or automation bias.

The hospital and clinical teams must set acceptable alert burden and minimum
useful lead time before threshold selection. The repository's default
`0.2` false alert events per nonseptic patient-day is a demonstration setting,
not an approved clinical limit. No single metric establishes clinical benefit.

## 3. Current-project boundary

The current CLI, saved model, and offline patient replay are suitable for
research and software testing. Keep them offline and label outputs as research
results. Do not send alerts to clinicians or patients, place scores in a
clinical record, or automate an intervention.

The existing source-B sample has been inspected in aggregate. It can still
support carefully described exploratory evaluation, but it is not an
untouched confirmatory holdout. Use a newly governed independent cohort for a
strict external confirmation. Treat sources A and B as public dataset cohorts,
not as verified identities of hospitals.

## 4. Data and target requirements

Before fitting a model for the candidate clinical target:

- Define the event using a documented clinical/data dictionary and specify
  exactly how event time is assigned. Have qualified reviewers assess whether
  the dataset-derived event time is suitable for the intended use.
- Construct eligible at-risk hours only before the event. Explicitly handle
  records positive at their first observation, label ambiguity, repeated
  events, discharge, death, and missing follow-up; do not infer precise onset
  from a censored record.
- Audit target timestamps, measurement timestamps, label availability, and
  feature availability for leakage. A feature at hour `t` may use only
  information available by hour `t`.
- Preserve patient/stay identity for grouping and splits, while excluding IDs,
  source, and administrative codes from predictors.
- Record data provenance and hashes. Keep raw and patient-level data out of
  GitHub and other unapproved systems.

If the proposed event cannot be derived and audited from available data, stop
the clinical-target work. Continue only with explicitly named challenge-label
benchmark experiments, without translating their scores into clinical risk.

## 5. Evaluation plan and release gates

### Gate 0 — Clinical question and protocol

Before model changes, obtain agreement from intended clinicians and hospital
operations on the population, event definition, prediction horizon, alert
recipient/action, acceptable alert burden, and outcome measures. Record these
as a versioned protocol. This repository cannot make those decisions on their
behalf.

**Exit:** signed-off protocol and an auditable target definition.

### Gate 1 — Retrospective development

Use the existing patient-level train/calibration/validation boundaries for
development. Fit learned preprocessing on training data, calibration on its
designated split, and select models and alert policies on validation only.
Keep the current threshold and cooldown as research parameters, not defaults
for care. Add patient-grouped bootstrap intervals, calibration reporting,
subgroup counts/results, event-level metrics, and missingness sensitivity
analyses.

**Exit:** reproducible results on the defined target, with no selection on
held-out outcomes.

### Gate 2 — Locked independent confirmation

Freeze the data/target specification, feature contract, model, calibration,
threshold, and analysis before accessing outcomes from a new external cohort.
Report site-specific and subgroup results, uncertainty, data-quality failures,
alert burden, and deviations. Do not retune on confirmation data.

**Exit:** independently reviewed evidence that meets the pre-specified
criteria. Failure means revise the research model and obtain a new holdout;
do not relabel a failed cohort as validation after inspecting it.

### Gate 3 — Prospective silent evaluation

Only after retrospective gates pass and approvals are obtained, run the system
in shadow mode: score live data without displaying alerts or changing care.
Evaluate data availability, missingness, calibration, alert volume, latency,
population shift, and subgroup behavior against the locked protocol.

**Exit:** operational and statistical acceptance criteria met, with a
documented monitoring and incident-response plan.

### Gate 4 — Human-factors and controlled pilot

Test the alert concept with intended users and the real workflow before any
care-facing pilot. Define who receives alerts, escalation and downtime
procedures, auditability, oversight, and stop criteria. Any clinical pilot
requires institutional, ethics, privacy, safety, and regulatory review as
applicable; it is outside the authority of this codebase.

**Exit:** approvals, trained users, reliable support and rollback, and a
prospective evaluation design able to assess benefit and harm.

## 6. Architecture implications

Keep the near-term system as a local, reproducible research pipeline. The
following changes support the gates above without implying readiness to serve
clinical alerts:

- Add a versioned **target-construction and eligibility** component separate
  from feature generation. Store its specification with every run.
- Preserve the canonical causal feature contract and explicit feature
  availability times. Do not let experiments silently redefine the target or
  feature logic.
- Make evaluation operate on patient/stay event episodes as well as hourly
  rows, report uncertainty and denominators, and distinguish hourly
  classifications from notification episodes.
- Freeze a run manifest containing input-data, target, split, feature,
  estimator, calibration, and alert-policy versions before held-out evaluation.
- Add site and subgroup summaries only where cohort size and governance allow;
  show counts and suppress unstable estimates rather than implying precision.
- Defer APIs, EHR integration, live alert delivery, and automated actions
  until the prior gates are passed and an operational owner and validated
  workflow exist.

## 7. Decisions still requiring clinical partners

This document deliberately does not choose on behalf of a hospital:

- the authoritative clinical event definition and its timestamp;
- the population and exclusions for a first deployment study;
- whether six hours is an appropriate and actionable horizon;
- what clinician or team receives an alert and what they should do;
- acceptable alerts per patient-day and minimum useful lead time;
- which subgroup analyses are clinically required and adequately powered;
- clinical, privacy, safety, ethics, and regulatory approvals.

Until these decisions are made and the gates are met, the project's valid
outcome is a reproducible research assessment of the public dataset—not a
claim that the model is ready to guide patient care.
