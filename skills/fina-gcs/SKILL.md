---
name: fina-gcs
description: Secure Google Cloud Storage administration and DuckDB S3 interoperability for RiskCube Parquet. Use for gcloud CLI operations, bucket/object inspection, Parquet reads, uploads, and deployment credential diagnostics.
---

# FInA GCS

Use this skill when working with the RiskCube GCS bucket, Parquet objects, DuckDB S3 interoperability, or low-level `gcloud` administration.

## Credential rules

Read `S3_API_KEY`, `S3_API_SECRET`, and `S3_BUCKET_NAME` from the environment. Never place credential values in source code, skills, prompts, scenario definitions, SQL text, logs, test fixtures, Git history, or user-facing output. Return only masked status such as whether credentials are present and the configured bucket URI.

The deployment environment is the source of truth for Vercel. Local development may use a permission-restricted, gitignored `.env.local`. Do not attach it or commit it. Set `S3_ENDPOINT` when using an S3-compatible endpoint other than `storage.googleapis.com`; the runtime defaults to `storage.googleapis.com`.

## DuckDB access

Configure DuckDB with the S3-compatible GCS endpoint:

```sql
CREATE OR REPLACE SECRET fina_gcs (
  TYPE S3,
  KEY_ID '${S3_API_KEY}',
  SECRET '${S3_API_SECRET}',
  ENDPOINT '${S3_ENDPOINT:-storage.googleapis.com}',
  URL_STYLE 'path',
  REGION 'auto'
);
```

Then query Parquet by object URI, for example `read_parquet('s3://fina-riskcube/coverage.parquet')`. Use the MCP `gcs_configuration_status` tool for non-secret diagnostics and `gcs_read_parquet` for configured-bucket reads. Prefer `read_parquet` with projection and filters rather than downloading objects into memory.

## gcloud CLI

Use low-level commands only after confirming the active project and account without printing tokens:

```bash
gcloud auth list
gcloud config get-value project
gcloud storage ls gs://<bucket>
gcloud storage objects list gs://<bucket>/<prefix>
gcloud storage cp <local-file> gs://<bucket>/<object>
gcloud storage cp gs://<bucket>/<object> <local-file>
```

The configured S3 interoperability key is not necessarily a gcloud OAuth credential. Do not run `gcloud auth activate-service-account` with it. Use gcloud for Google IAM/OAuth-authenticated administration and DuckDB S3 secrets for the interoperability path. Report the endpoint from `S3_ENDPOINT` in masked diagnostics, but never report key or secret values.

## Object layout

Prefer integer-key, versioned, scenario-partitioned objects such as:

```text
gs://fina-riskcube/version_id=<version_id>/scenario_id=<scenario_id>/instance_id=<instance_id>.parquet
```

Persist the human-readable `version_key` and `scenario_key` in the DuckDB catalogs and Parquet columns, but use integer IDs in object paths for compactness and partition pruning. Keep immutable partitions and write a new instance for retries or recalculation. Validate object existence and row counts after uploads. Preserve integer and human-readable identifiers in downstream results.
