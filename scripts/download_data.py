"""Download a seeded subset, or all records, from the official open dataset.

    python scripts/download_data.py --limit 500 --output data/physionet

The default downloads 500 patients per hospital. --limit 0 downloads all.
The download reads data only; it does not import or execute remote code. The logic lives in
src/sepsis_warning/download.py so the SageMaker `prepare` job can use it too.
"""

from sepsis_warning.download import main

if __name__ == "__main__":
    main()
