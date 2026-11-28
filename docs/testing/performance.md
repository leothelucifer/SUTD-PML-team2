# AWS API performance test

Run the manual `performance` GitHub Actions workflow after deployment and
verification. It downloads a seeded public-data sample, prepares at most 50
anonymized Hospital B stays in the runner's temporary directory, verifies the
API has a non-synthetic model, then sends bounded SigV4-signed Locust requests
to `POST /predict`.

The load is capped at 10 users and 120 seconds. The payload JSONL and source
records are not uploaded or committed. The workflow artifact contains only
aggregate request statistics, model provenance, and a summary; download it
with `gh run download <run-id>` and retain only those aggregate results.

Run the local payload and report tests with:

```bash
python -m unittest discover -s tests -p test_performance_tools.py -v
```

The performance dependency lock is regenerated with
`bash scripts/lock_requirements.sh`. See the [workflow reference](../WORKFLOWS.md)
for manual inputs and the [AWS run order](../RUN-ORDER.md) for deployment,
monitoring, results collection, and teardown.
