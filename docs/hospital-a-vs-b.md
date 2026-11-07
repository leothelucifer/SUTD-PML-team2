# Hospital A vs hospital B: what differs

Full public cohorts: 20,336 ICU stays from A and 20,000 from B (PhysioNet
Challenge 2019, v1.0.0). Descriptive only: nothing here is used to choose
features, models or thresholds, and B stays the untouched external test set.
All numbers are aggregates.

Recreate every figure and the table:

```bash
python -m sepsis_warning compare-hospitals --data data/physionet/combined.parquet --output reports/figures
```

Figures and the full summary table: [reports/figures/](../reports/figures/README.md).

## Headline findings (slide 5)

1. **B has fewer sepsis cases.** 5.7% of B stays are septic vs 8.8% at A, and
   1.4% vs 2.2% of hours carry a positive label. A model trained on A will
   see a different base rate at B, so precision and alert counts will change
   even if the model's ranking is just as good
   ([fig 1](../reports/figures/01_cohort_prevalence.png)).
2. **The hospitals measure different things.** The biggest gap is in
   missingness, not in the values themselves. Blood-gas labs are almost
   never recorded at B (BaseExcess, HCO3, pH, FiO2, Chloride: 98-100%
   missing vs 86-92% at A). Resp is missing in 21% of B hours vs 10% at A.
   DBP is the other way round (48% missing at A vs 14% at B). Features built
   on "was this measured?" may not transfer
   ([fig 4](../reports/figures/04_missingness.png)).
3. **Most vitals look alike, with one exception.** HR, Temp and Resp have
   almost the same distribution, but MAP is clearly higher at B (median 84
   vs 77 mmHg). B also mostly charts Resp in even numbers (16, 18, 20),
   which shows a different charting habit
   ([fig 5](../reports/figures/05_vital_signs.png)).
4. **SIRS fires much more at A in patients without sepsis** (33% vs 24% of
   their hours), but about the same in septic patients (47% vs 44%). Since
   SIRS is a fixed rule, this gap comes from the patients and the recording
   practices, not from overfitting
   ([fig 6](../reports/figures/06_sirs_firing_rate.png)).

Similar at both: stay length (median 39 vs 38 h, almost all stays 8-60 h),
age (median 65 vs 62 years) and recorded gender (58% vs 54% male)
([fig 2](../reports/figures/02_stay_length.png),
[fig 3](../reports/figures/03_age_gender.png)).

## Points for the report

- **More B septic stays are positive from their first row** (19.5% vs 11.3%).
  Their onset is unknown (sepsis began before the ICU record), so they are
  excluded from lead-time analysis. At B a larger share of septic patients
  cannot be warned early at all
  ([fig 7](../reports/figures/07_label_timing.png)).
- EtCO2 is 100% missing at A but recorded in about 8% of B hours. A model
  trained on A has never seen it.
- B is 3 years younger on median age and has fewer patients aged 80+ (11.6% vs
  15.0%). This matters for the age-group fairness checks.
- A record is an ICU stay, not a person. The same person could appear more
  than once.
