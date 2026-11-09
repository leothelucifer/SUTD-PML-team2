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
| Second DVC remote | Private SUTD SharePoint site, reached through a synced folder and set in `.dvc/config.local` | No |
| Hashes and pipeline steps | `dvc.yaml`, `dvc.lock` | Yes |

**Access:** only Edwin can reach these remotes for now: one is a folder on one laptop, and the other is a private SharePoint site with no other members.
No credentials are involved.

**Redistribution:** the data is not shared with anyone else. PhysioNet's project page and the
dataset's `LICENSE.txt` state different licences (risk R14), so any shared remote
should wait until that is settled. Teammates get the data from PhysioNet using the
steps in the README.

DVC's anonymous usage reporting is turned off in `.dvc/config`.

## Install

```bash
python -m pip install -e ".[dvc]"
```

## Restore into a fresh checkout (only with access to the private folder)

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
| `data/physionet/splits.csv` | `c71b3b34f59a0bf0f5de724d94de211c65f0fee83d0f72a712fa6b77f358c800` |

`dvc.lock` also records the raw folders: 20,336 files for hospital A and 20,000
for hospital B. The raw files are not stored in the remote, as they can be
downloaded again from PhysioNet.

## Restore from the SharePoint copy (once you have been added)

Members will be added only after the licence question (R14) is settled. Until then,
please use the rebuild steps below.

1. Open the SharePoint site, go to **Documents**, click **Sync**, and allow OneDrive to open.
2. In File Explorer, right-click the `dvc-storage` folder and choose **Always keep on this device**.
3. Point DVC at your own synced folder and restore the files:

```bash
python -m pip install -e ".[dvc]"
dvc remote add -d --local shared "<path to your synced dvc-storage folder>"
dvc pull
```

The path is different on every computer, so it stays in the Git-ignored
`.dvc/config.local`. Please do not rename or delete anything inside `dvc-storage`.

## Rebuild the same files on your own machine

The DVC remote is a private folder on one laptop, so `dvc pull` will not work for
anyone else. Rebuilding from PhysioNet gives the same files without sharing any data.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[checks,dvc]"
python scripts/download_data.py --limit 0 --output data/physionet
python -m sepsis_warning prepare --data data/physionet --output data/physionet/combined.parquet
python -m sepsis_warning split --data data/physionet/combined.parquet --output data/physionet/splits.csv
python -m sepsis_warning validate
```

You should see 40,336 stays and 1,552,210 rows. Then compare both files with the
SHA-256 values in the table above:

```bash
sha256sum data/physionet/combined.parquet data/physionet/splits.csv
```

On Windows Command Prompt or PowerShell, use `certutil -hashfile <file> SHA256`.

- Use the versions in `requirements.txt`, because a different Parquet library
  version can write different bytes from the same data.
- Compare with SHA-256 rather than `dvc status`. `dvc.lock` was recorded on Windows,
  so `dvc status` can report a change on another system when only line endings differ.
- Please do not run `dvc push` or add a shared remote while risk R14 is open.

## After changing the data

`dvc repro` re-runs the steps whose inputs changed, then `dvc push` stores the new
outputs. `dvc repro` replaces the old outputs, so copy anything you need to keep first.
