# Hospital A vs B figures

Descriptive only: nothing here is used to choose models or thresholds.
Hospital A = its TRAINING split only (11,183 stays); A's calibration,
validation and internal-test stays are left out. Hospital B = all 20,000
stays, described only. All values are aggregates over stays or hourly rows.

Recreate everything in this folder:

    python -m sepsis_warning compare-hospitals --data data/physionet/combined.parquet --splits data/physionet/splits.csv --output reports/figures

| File | Shows |
|---|---|
| 01_cohort_prevalence.png | Stays, septic stays, septic share and positive-hour share |
| 02_stay_length.png | Stay length distribution (log scale) |
| 03_age_gender.png | Age groups and recorded gender |
| 04_missingness.png | % of hours with no measurement, per variable |
| 05_vital_signs.png | HR, MAP, Temp, Resp distributions (measured hours) |
| 06_sirs_firing_rate.png | % of hours with SIRS >= 2, septic vs non-septic stays |
| 07_label_timing.png | ICU hour of the first positive label |
| 08_sirs_threshold_tradeoff.png | SIRS at >= 1..4 criteria, validation only (made by `python -m sepsis_warning baseline --data data/physionet --output artifacts/baseline`) |
| hospital_summary.csv | Summary table below |
| missingness_by_hospital.csv | Missing % for all 34 hourly variables |

## Summary table

| | Hospital A | Hospital B |
|---|---:|---:|
| ICU stays | 11,183 | 20,000 |
| Hourly rows | 435,449 | 761,995 |
| Septic stays | 984 | 1,142 |
| Septic stays (%) | 8.8 | 5.7 |
| Positive-label hours (%) | 2.2 | 1.4 |
| Stay length, median h (IQR) | 39 (25-47) | 38 (23-47) |
| Septic stays positive from first hour (%) | 12.1 | 19.5 |
| Age, median (years) | 64.7 | 62.0 |
| Male (%) | 58.0 | 53.7 |
| Hours with SIRS >= 2, all (%) | 34.7 | 26.2 |
| Hours with SIRS >= 2, septic stays (%) | 46.7 | 44.1 |
| Hours with SIRS >= 2, non-septic stays (%) | 32.9 | 24.4 |
| HR missing (%) | 7.8 | 12.1 |
| Temp missing (%) | 66.2 | 66.1 |
| WBC missing (%) | 92.5 | 94.7 |
| Lactate missing (%) | 96.5 | 98.1 |
