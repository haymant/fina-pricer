# High-throughput RiskCube storage architecture

The storage layer separates **scenario definitions**, **execution coordination**, and **materialized risk cells**. DuckDB is used as the in-memory orchestration and OLAP catalog. Parquet is used for append-only, compressed RiskCube partitions that can be queried by DuckDB, Python, or other analytical engines.

## Logical tables

| Table | Purpose | Key fields |
|---|---|---|
| `scenario_catalog` | Stores materialized snapshots or rule-based scenario definitions | `scenario_id`, `scenario_version`, `scenario_kind`, `materialization_mode` |
| `riskcube_instances` | Coordinates a batch/version/scenario execution | `instance_id`, `batch_id`, `version`, `scenario_id`, `status` |
| `riskcube_partitions` | Maps a completed instance to its Parquet partition | `partition_id`, `instance_id`, `version`, `scenario_id`, `parquet_path` |
| `riskcube_cells` | In-memory flat analytical table for current or recent results | scenario/version coordinates, RFK axes, valuation fields, sensitivities |

`riskcube_instances.scenario_id` is the foreign-key relationship to the scenario catalog. `riskcube_partitions.instance_id` links persisted materialized cells back to the execution instance.

## Scenario modes

A scenario can store a complete `market_data_snapshot`, a rule set in `market_data_manipulations`, or both. Rule-based scenarios are materialized at execution time against each request. The attached builder supports spot, volatility, FX, interest-rate, dividend, and FX-volatility manipulations, together with filter conditions. `ScenarioBuilder.current_report()` creates a virtual current-market report template; a market-data resolver can populate its snapshot at execution time.

## Partitioning

Completed cells are written to paths of the form:

```text
<root>/version=<version>/scenario_id=<scenario_id>/instance_id=<instance_id>.parquet
```

This makes version and scenario natural partition-pruning keys. When `S3_BUCKET_NAME` or `RISKCUBE_PARQUET_ROOT` is an `s3://` or `gs://` URI, DuckDB writes the partition directly to that remote root using the configured S3 secret. On Vercel, if no remote root is configured, the service uses `/tmp/riskcube` only as an ephemeral fallback; it never attempts to create a read-only project-directory path. The instance remains the execution-level unit, allowing retries and multiple batches for the same scenario/version without overwriting historical materializations.

## Flat cell contract

Every cell stores the canonical RFK object and a flat `coordinates` projection whose keys match `RiskCube.axes`:

```text
type, underlying, currency_pair, expiry, strike, tenor,
temporal_role, date, surface_parameter
```

The cell also contains `pv_amount`, `price_pct_of_notional`, `method`, `bump`, JSON sensitivity fields, and explainability/valuation JSON. This supports two views of the same result:

1. **Relational/OLAP view:** scalar axis and valuation columns for DuckDB predicates, aggregation, and AG Grid server-side filtering.
2. **AI analysis view:** reconstructed nested JSON grouped by `instance_id`, `scenario_id`, `case_id`, and instrument.

A typical OLAP query is:

```sql
SELECT scenario_id, underlying, type, AVG(CAST(sensitivities_json->>'delta' AS DOUBLE)) AS avg_delta
FROM read_parquet('s3://fina-riskcube/version=*/scenario_id=*/*.parquet')
WHERE version = 'v1'
GROUP BY scenario_id, underlying, type;
```

The current implementation intentionally keeps JSON copies of RFK, coordinates, sensitivities, explainability, and valuation in addition to scalar columns. This provides provenance and lossless reconstruction while preserving fast columnar access for common dimensions.

## Execution flow

The batch executor registers a scenario, creates a `riskcube_instances` row with `RUNNING` status, materializes each request, calls the pricing engine, inserts flattened cells into DuckDB, writes one Parquet partition, and marks the instance `COMPLETED`. The same design can later be extended with a queue or parallel worker pool without changing the storage contract.
