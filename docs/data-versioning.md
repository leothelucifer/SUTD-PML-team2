# Data versioning with DVC (E02)

DVC records which data files belong with this code. Git holds only `dvc.yaml`,
`dvc.lock` and `.dvc/config`, which contain sizes and hashes. Patient data, the
DVC cache and any credentials stay out of Git.

## Where the data lives

| Item | Where | In Git? |
|---|---|---|
| Raw `.psv` files, `combined.parquet`, `splits.csv` | Git-ignored `data/` folder | No |
| DVC cache | `.dvc/cache` | No |
| DVC remote | Private folder on Edwin's laptop, set in `.dvc/config.local` | No |
| Hashes and pipeline steps | `dvc.yaml`, `dvc.lock` | Yes |

**Access:** only Edwin can reach this remote, because it is a folder on one laptop.
No credentials are involved.

**Redistribution:** nothing is uploaded or shared. PhysioNet's project page and the
dataset's `LICENSE.txt` state different licences (risk R14), so any shared remote
should wait until that is settled. Teammates get the data from PhysioNet using the
steps in the README.

DVC's anonymous usage reporting is turned off in `.dvc/config`.

## Install

```bash
python -m pip install -e ".[dvc]"
```

## Restore into a fresh checkout

```bash
git clone <repo> fresh-copy
cd fresh-copy
dvc remote add -d --local localstore <path to the private DVC folder>
dvc pull
```

Check the restored files against these SHA-256 values:

| File | SHA-256 |
|---|---|
| `data/physionet/combined.parquet` | `85f044951122bff3a18adaa1a5b8e86b620a752e14e6553b4c241b10706bba85` |
| `data/physionet/splits.csv` | Listed as `splits_csv_sha256` in `splits.csv.summary.json`; begins `c71b3b34` |

`dvc.lock` also records the raw folders: 20,336 files for hospital A and 20,000
for hospital B. The raw files are not stored in the remote, as they can be
downloaded again from PhysioNet.

## After changing the data

`dvc repro` re-runs the steps whose inputs changed, then `dvc push` stores the new
outputs. `dvc repro` replaces the old outputs, so copy anything you need to keep first.
