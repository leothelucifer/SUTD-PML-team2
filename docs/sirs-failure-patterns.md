# Where the SIRS rule fails

Group totals from hospital A's **validation split only** (3,051 ICU stays, 268
septic). The internal test and hospital B are not used, because searching for
patterns is a form of tuning. No patient-level rows are reported, and any group
with fewer than 10 stays is blanked. The rule stays fixed at 2 or more of 4
criteria. PhysioNet Challenge 2019, v1.0.0.

Recreate the tables (written to the git-ignored `artifacts/baseline/failures/`):

```bash
python -m sepsis_warning sirs-failures --data data/physionet/combined.parquet --output artifacts/baseline/failures
```

## False alarms: common signs of being ill, without infection

SIRS flagged 40,107 hours that carry no sepsis label.

| Criteria met | False-alarm hours | Share |
|---|---|---|
| Heart rate + breathing | 8,059 | 20.1% |
| Breathing + white cells | 7,237 | 18.0% |
| Heart rate + white cells | 6,995 | 17.4% |
| Heart rate + breathing + white cells | 6,978 | 17.4% |
| Any combination that includes temperature | 10,838 | 27.0% |

Fast heart rate, fast breathing and abnormal white cells are common in ICU
patients, so about 73% of false alarms come from those three signs alone.

## Missed septic hours: too few signs, more often than missing tests

Of 2,578 septic-labelled hours, SIRS caught 1,381 and missed 1,197.

| Sign | Met when caught | Met when missed | Never measured when caught | Never measured when missed |
|---|---|---|---|---|
| Temperature | 36.1% | 7.7% | 1.5% | 7.1% |
| Heart rate | 74.1% | 14.6% | 0.0% | 1.4% |
| Breathing | 73.2% | 20.0% | 0.0% | 1.4% |
| White cells | 63.9% | 19.6% | 8.6% | 20.7% |

An unmeasured value counts as normal, so white cells never having been measured
explains part of the misses (20.7% of missed hours, against 8.6% of caught hours).
Most missed hours were measured and simply did not look abnormal enough.

## Septic stays with no warning before onset

| Outcome | Stays | Median hours before onset | White cells seen before onset | Median peak signs in the 12 h before onset |
|---|---|---|---|---|
| Warned | 190 | 35 | 92.6% | 3 |
| Not warned | 54 | 43 | 83.3% | 1 |

These are the 244 septic stays whose onset can be placed (same definition as
early detection, 190 / 244 = 0.78). Stays positive from their first row are left
out because their onset is unknown. The stays not warned usually reached only
one sign, even though 83% had a white-cell result.

## By time in the ICU

| ICU hour | Stays | Precision | Recall | Specificity | Share of hours flagged |
|---|---|---|---|---|---|
| 1-6 | 2,966 | 0.024 | 0.395 | 0.778 | 22.4% |
| 7-24 | 3,050 | 0.026 | 0.517 | 0.658 | 34.5% |
| 25-48 | 2,376 | 0.015 | 0.561 | 0.627 | 37.5% |
| 49+ | 735 | 0.124 | 0.570 | 0.557 | 45.6% |

Recall is lowest in the first 6 hours, when few readings exist. Precision is
highest after hour 48, where septic-labelled hours are far more common.
