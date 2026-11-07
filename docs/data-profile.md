# Understanding the PhysioNet Challenge 2019 data

This document describes the local sample currently used by the project and the
meaning of its columns. It is an exploratory profile, not a clinical
interpretation or a claim about the full PhysioNet cohort.

## Source and local sample

The source is the [PhysioNet/Computing in Cardiology Challenge 2019
dataset](https://physionet.org/content/challenge-2019/1.0.0/), version 1.0.0.
It contains one pipe-delimited (`.psv`) file per ICU record. Each row represents
one recorded ICU hour. Source A and source B are the two publicly available
cohorts; they should be treated as data-source cohorts, not as identifiable
hospital names. The full public training set contains 40,336 records: 20,336
from A and 20,000 from B. The local sample below is only a subset.

The local download was generated with seed 42 and a limit of 500 records per
source:

```bash
python scripts/download_data.py --limit 500 --output data/physionet
python -m sepsis_warning prepare \
  --data data/physionet \
  --output data/physionet/combined.parquet
```

The first command writes a `download_manifest.json` containing the source,
seed, selected record paths, and SHA-256 hashes. The second validates the PSV
files and combines them into a local Parquet file. The data and manifest belong
in the Git-ignored `data/` directory, not in this repository.

**Observed local snapshot (2026-11-07):**

| Source | ICU records | Hourly rows | Records with any positive label | Positive-label hours | Median hours per record (IQR) |
|---|---:|---:|---:|---:|---:|
| A | 500 | 19,062 | 37 (7.40%) | 355 (1.86%) | 37 (24–47) |
| B | 500 | 21,038 | 28 (5.60%) | 269 (1.28%) | 39 (24.75–48) |
| **Total** | **1,000** | **40,100** | **65 (6.50%)** | **624 (1.56%)** | — |

Record lengths in this sample range from 8 to 336 hours. A record is an ICU
stay and is not guaranteed to correspond to a unique person. The download is a
seeded subset of the available records, not the complete dataset; the counts
and percentages above must not be reported as full-cohort prevalence estimates.
The hourly percentage is row-weighted, so longer stays contribute more.

The combined Parquet has 43 columns: the 40 original time-dependent input
variables, `SepsisLabel`, and project-added `patient_id` and `hospital`
provenance columns. `patient_id` is the PSV filename stem used as a grouping
key, not a model feature.

## Row structure and target

Each `.psv` row contains measurements for one ICU hour. Missing measurements
are encoded as `NaN`; missing does not mean zero or normal. `ICULOS` is the
hour index since ICU admission, and the project verifies that it increases by
one within each record. `HospAdmTime` is the number of hours between hospital
admission and ICU admission; negative values are expected when hospital
admission precedes ICU admission.

`SepsisLabel` is the challenge target, not a feature. For sepsis records, it is
already positive from six hours before the challenge-defined sepsis onset and
remains positive afterward. It is zero throughout non-sepsis records. Do not
shift it again. A model trained against this label estimates the supplied
benchmark label; it does not automatically estimate a clinically validated
future risk. In the local sample, positive labels occur in 1.56% of hourly rows
(355/19,062 in A and 269/21,038 in B). Record prevalence is a different
quantity (6.50% overall).

Preparation checks the required schema, numeric values, absence of infinities,
binary nondecreasing labels, and consecutive ICU-hour indices before writing
the combined file. These checks do not prove that every value is clinically
plausible.

## Input variables

The challenge defines 40 time-dependent input columns. The variable meanings
and units below follow the [official data description](https://physionet.org/content/challenge-2019/1.0.0/).

### Vital signs

| Column | Meaning | Unit |
|---|---|---|
| `HR` | Heart rate | beats/min |
| `O2Sat` | Pulse oximetry | % |
| `Temp` | Temperature | °C |
| `SBP` | Systolic blood pressure | mmHg |
| `MAP` | Mean arterial pressure | mmHg |
| `DBP` | Diastolic blood pressure | mmHg |
| `Resp` | Respiration rate | breaths/min |
| `EtCO2` | End-tidal carbon dioxide | mmHg |

### Laboratory measurements

| Column | Meaning | Unit |
|---|---|---|
| `BaseExcess` | Base excess | mmol/L |
| `HCO3` | Bicarbonate | mmol/L |
| `FiO2` | Fraction of inspired oxygen | % / fraction as recorded |
| `pH` | Blood pH | unitless |
| `PaCO2` | Arterial carbon-dioxide partial pressure | mmHg |
| `SaO2` | Arterial oxygen saturation | % |
| `AST` | Aspartate transaminase | IU/L |
| `BUN` | Blood urea nitrogen | mg/dL |
| `Alkalinephos` | Alkaline phosphatase | IU/L |
| `Calcium` | Calcium | mg/dL |
| `Chloride` | Chloride | mmol/L |
| `Creatinine` | Creatinine | mg/dL |
| `Bilirubin_direct` | Direct bilirubin | mg/dL |
| `Glucose` | Serum glucose | mg/dL |
| `Lactate` | Lactic acid | mg/dL |
| `Magnesium` | Magnesium | mmol/dL |
| `Phosphate` | Phosphate | mg/dL |
| `Potassium` | Potassium | mmol/L |
| `Bilirubin_total` | Total bilirubin | mg/dL |
| `TroponinI` | Troponin I | ng/mL |
| `Hct` | Hematocrit | % |
| `Hgb` | Hemoglobin | g/dL |
| `PTT` | Partial thromboplastin time | seconds |
| `WBC` | Leukocyte count | 10³/µL |
| `Fibrinogen` | Fibrinogen | mg/dL |
| `Platelets` | Platelet count | 10³/µL |

### Demographics, administrative fields, and time

| Column | Meaning | Project treatment |
|---|---|---|
| `Age` | Age in years; 100 represents age 90 or above in the challenge data | Included |
| `Gender` | Female = 0, male = 1 in the challenge description | Included |
| `HospAdmTime` | Hours from hospital admission to ICU admission | Included |
| `ICULOS` | ICU-hour index | Included |
| `Unit1` | Administrative ICU-unit indicator (MICU) | Excluded from model features |
| `Unit2` | Administrative ICU-unit indicator (SICU) | Excluded from model features |

The project constructs model inputs from 34 physiological variables
(`DYNAMIC`) and the four fields `Age`, `Gender`, `HospAdmTime`, and `ICULOS`
(`STATIC`). `Unit1` and `Unit2` are deliberately excluded because they are
administrative identifiers that can encode unit-specific care or measurement
practice. `patient_id`, `hospital`, and `SepsisLabel` are also not model
features.

## Missingness and source differences

Missingness is expected and highly variable across measurements. The table
shows the percentage of missing hourly values in the 40,100-row local sample;
these are row-level rates, not the percentage of records with a measurement.

| Variable | Missing | Variable | Missing |
|---|---:|---|---:|
| `HR` | 9.78% | `O2Sat` | 13.26% |
| `Temp` | 66.70% | `SBP` | 15.11% |
| `MAP` | 12.80% | `DBP` | 31.37% |
| `Resp` | 15.51% | `EtCO2` | 94.64% |
| `BaseExcess` | 95.13% | `HCO3` | 96.13% |
| `FiO2` | 92.55% | `pH` | 93.56% |
| `PaCO2` | 94.87% | `SaO2` | 96.65% |
| `AST` | 98.40% | `BUN` | 93.26% |
| `Alkalinephos` | 98.41% | `Calcium` | 94.10% |
| `Chloride` | 95.83% | `Creatinine` | 94.00% |
| `Bilirubin_direct` | 99.80% | `Glucose` | 82.97% |
| `Lactate` | 97.53% | `Magnesium` | 93.88% |
| `Phosphate` | 96.12% | `Potassium` | 90.92% |
| `Bilirubin_total` | 98.54% | `TroponinI` | 99.10% |
| `Hct` | 91.57% | `Hgb` | 92.78% |
| `PTT` | 97.41% | `WBC` | 93.65% |
| `Fibrinogen` | 99.48% | `Platelets` | 94.19% |
| `Age` | 0.00% | `Gender` | 0.00% |
| `HospAdmTime` | 0.00% | `ICULOS` | 0.00% |
| `Unit1` | 39.44% | `Unit2` | 39.44% |

Missingness patterns differ between sources in this sample. For example, `DBP`
is missing in 51.31% of A rows and 13.31% of B rows; `Resp` is missing in
10.33% of A rows and 20.19% of B rows; `EtCO2` is missing in all sampled A rows
but 89.79% of sampled B rows. These observed differences may reflect
measurement and documentation practices as well as case mix. They are a reason
to report missingness and test transfer robustness, not evidence by themselves
of a clinical or data-quality defect.

Keep missing values missing in descriptive analyses. For modeling, use only
past/current information, forward-fill within each record if justified, and
fit any imputation fallback on training records only. Do not backfill from
future hours or interpret missingness as a normal result.

## Quality review and modeling implications

The local preparation step succeeded for all 1,000 downloaded records and
40,100 hourly rows. Its record-level checks found the expected consecutive
hours and nondecreasing binary labels. The following points still need care:

- The sample is outcome-imbalanced: 1.56% of hourly labels are positive.
  Accuracy alone is misleading; report precision-recall metrics and
  record-level outcomes as well.
- Observed extremes should be reviewed in context, not automatically removed
  or clipped. For example, the sample includes high values such as glucose
  912 mg/dL, AST 9,840 IU/L, and WBC 168.6 × 10³/µL. This profile does not
  determine whether an individual value is erroneous.
- Stay length varies substantially. Use record-grouped splits and uncertainty
  estimates so that rows from the same stay never appear across train and test.
- Source-specific missingness may let a model learn measurement workflows.
  Compare performance and missingness behavior across A and B; do not use
  `Unit1`, `Unit2`, patient IDs, or source labels as shortcuts.
- The project uses A for train/calibration/validation/internal test and holds B
  out for external evaluation. Freeze model and alert-policy decisions before
  evaluating B.
- In the complete-data experiment, split A at the record level into
  approximately 55% train, 15% calibration, 15% validation, and 15% internal
  test. Use training records to fit models and preprocessing, calibration
  records to calibrate probabilities, and validation records to select models
  and thresholds. Do not tune against internal-test or B results.
- At hour *t*, use only current and past observations. Do not backfill, use
  whole-stay summaries, or include labels, patient IDs, source IDs, or
  administrative unit codes as predictors. Prefix-invariance checks help catch
  future leakage.
- Report more than accuracy: include average precision, AUROC, Brier score or
  calibration, and the project's hourly utility and alert-burden metrics.
  Separate hourly positive predictions from cooldown-suppressed notification
  events. Interpret inferred warning lead time cautiously, especially for
  records already positive at their first recorded hour.
- This public research dataset and project prototype do not establish clinical
  validity or support treatment decisions.

The benchmark rewards predictions in a timing window around a challenge-defined
onset; it does not demonstrate that a model is clinically useful. Use grouped
uncertainty estimates because hours within a record are correlated, and report
source-specific performance rather than treating all hourly rows as independent.

## Citation and data handling

When reporting or redistributing derived work, follow the dataset's
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) and cite:

> Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B.,
> Sharma, A., Nemati, S., & Clifford, G. D. (2019). *Early Prediction of Sepsis
> from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*
> (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857

Raw PSV files, combined Parquet, download manifests, and patient-level
visualizations must remain local under the Git-ignored `data/` directory.
Inspect notebook outputs before sharing or committing them.
