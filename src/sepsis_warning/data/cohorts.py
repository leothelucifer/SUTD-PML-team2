"""Shared patient-cohort loading for baseline and training workflows."""


from .records import load_patients, split_patients
from .splits import load_split_patients


def load_cohorts(args):
    """Return (cohorts, synthetic) for `baseline` and `sirs-failures`.

    A Parquet file as --data uses the saved splits.csv. A folder of .psv files
    still reads every record and re-splits with --seed.
    """
    if args.data.is_file():
        splits_path = args.splits or args.data.with_name("splits.csv")
        if not splits_path.exists():
            raise ValueError(f"{splits_path} not found; run `split` first")
        cohorts = load_split_patients(args.data, splits_path)
        ids = [p.patient_id for cohort in cohorts.values() for p in cohort]
        flagged = sum("synthetic_" in patient_id for patient_id in ids)
        if flagged not in (0, len(ids)):
            raise ValueError(f"{args.data} mixes synthetic and real records")
        synthetic = bool(flagged)
    else:
        synthetic = (args.data / "SYNTHETIC.json").exists()
        patients = load_patients(args.data)
        if not synthetic and any("synthetic_" in p.patient_id for p in patients):
            raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
        cohorts = split_patients(patients, args.seed)
    print("Data mode:", "SYNTHETIC SOFTWARE FIXTURE" if synthetic else "PhysioNet records", flush=True)
    return cohorts, synthetic
