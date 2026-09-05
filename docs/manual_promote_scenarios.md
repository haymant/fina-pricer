# Manual MCP verification: multi-scenario RiskCube promotion

This runbook verifies that scenario definitions are persisted, that one release version can coordinate several scenarios, and that each scenario produces an immutable RiskCube instance and integer-keyed Parquet partition.

## 1. Inspect the durable catalogs

Call the following MCP tools first:

```text
version_list()
scenario_list()
gcs_configuration_status()
```

Expected checks:

| Check | Expected result |
|---|---|
| Version catalog | A list of entries with integer `version_id`, stable `version_key`, and `created_at` |
| Scenario catalog | A list of entries with integer `scenario_id`, stable `scenario_key`, definition type, and source timestamps |
| GCS status | Credentials are reported only as present/missing; no key or secret value is returned |

If the catalogs are empty, continue with the creation prompts below. If a key already exists, use a new key rather than changing a completed production definition.

## 2. Create a base scenario

Call `scenario_create` with:

```json
{
  "scenario_id": "PROMO_BASE_20260905",
  "scenario_name": "Promotion base market",
  "scenario_version": "1",
  "description": "Base market snapshot for the FCN promotion verification",
  "scenario_kind": "rule",
  "materialization_mode": "rules",
  "base_market_data_datetime": "2026-09-05T00:00:00Z",
  "trade_repository_snapshot_datetime": "2026-09-05T00:00:00Z",
  "market_data_manipulations": []
}
```

The response should contain both `scenario_id` as an integer surrogate and `scenario_key` equal to `PROMO_BASE_20260905`. Call `scenario_get` with either the returned integer or the stable key and confirm that the definition is readable after creation.

## 3. Create two stress scenarios

Create a downside spot scenario:

```json
{
  "scenario_id": "PROMO_SPOT_DOWN_10_20260905",
  "scenario_name": "Ten percent spot-down stress",
  "scenario_version": "1",
  "description": "Ten percent relative spot shock applied to the selected basket underlyings",
  "scenario_kind": "rule",
  "materialization_mode": "rules",
  "base_market_data_datetime": "2026-09-05T00:00:00Z",
  "trade_repository_snapshot_datetime": "2026-09-05T00:00:00Z",
  "market_data_manipulations": [
    {
      "type": "spot_shift",
      "shift_type": "percentage",
      "value": -0.10
    }
  ]
}
```

Create a volatility stress scenario:

```json
{
  "scenario_id": "PROMO_VOL_UP_500BP_20260905",
  "scenario_name": "Volatility up stress",
  "scenario_version": "1",
  "description": "Volatility stress for sensitivity and barrier-hit analysis",
  "scenario_kind": "rule",
  "materialization_mode": "rules",
  "base_market_data_datetime": "2026-09-05T00:00:00Z",
  "trade_repository_snapshot_datetime": "2026-09-05T00:00:00Z",
  "market_data_manipulations": [
    {
      "type": "vol_manipulation",
      "shift_type": "absolute",
      "value": 0.05,
      "underlying_name": "AAPL"
    },
    {
      "type": "vol_manipulation",
      "shift_type": "absolute",
      "value": 0.05,
      "underlying_name": "TSLA"
    }
  ]
}
```

Call `scenario_list()` again. Confirm that all three definitions are present and that their integer IDs are distinct and stable.

## 4. Prepare one request set for all scenarios

Use the same request set for every scenario so differences are attributable to the scenario definition rather than to instrument population. The request array must contain one object per FCN instrument:

```json
[
  {"case_id": "FCN_AAPL_TSLA_001", "request": {"...": "complete FCN pricing request"}},
  {"case_id": "FCN_AAPL_TSLA_002", "request": {"...": "complete FCN pricing request"}},
  {"case_id": "FCN_SINGLE_UNDERLYING_003", "request": {"...": "complete FCN pricing request"}}
]
```

The real payload must include the complete instrument key, unwind map, market snapshot, coupon, maturity, observation schedule, independent strikes, and independent relative barriers. Do not send credentials in the request.

## 5. Promote all scenarios under one version

Use one immutable release key and one promotion batch ID. Call `scenario_trigger` once per scenario. The calls may be submitted sequentially during manual verification; production workers may execute them concurrently.

Base scenario:

```json
{
  "scenario_id": "PROMO_BASE_20260905",
  "version": "risk-release-20260905",
  "batch_id": "promotion-20260905-fcn-basket",
  "requests": [
    {"case_id": "FCN_AAPL_TSLA_001", "request": {"...": "complete request"}},
    {"case_id": "FCN_AAPL_TSLA_002", "request": {"...": "complete request"}},
    {"case_id": "FCN_SINGLE_UNDERLYING_003", "request": {"...": "complete request"}}
  ]
}
```

Repeat the same call shape with `scenario_id` set to `PROMO_SPOT_DOWN_10_20260905` and then `PROMO_VOL_UP_500BP_20260905`.

For every response, record:

```text
instance_id
batch_id
version_id
version_key
scenario_id
scenario_key
instrument_count
cell_count
partitions
```

Expected invariants are that all three responses have the same `version_id` and `version_key`, different `scenario_id` values, different `instance_id` values, the same instrument count, and partition paths of the form:

```text
s3://fina-riskcube/version_id=<integer>/scenario_id=<integer>/instance_id=<uuid>.parquet
```

## 6. Verify materialized cells with OLAP

Use `olap_query` to inspect the catalog-to-instance mapping:

```sql
SELECT version_id, version_key,
       scenario_id, scenario_key,
       instance_id, status,
       instrument_count, cell_count
FROM riskcube_instances
WHERE batch_id = 'promotion-20260905-fcn-basket'
ORDER BY scenario_id;
```

Verify that every row is `COMPLETED` and that there is one instance per scenario.

Compare instrument-level PV and delta across scenarios:

```sql
SELECT scenario_id, scenario_key, instrument_id,
       SUM(pv_amount) AS pv_amount,
       SUM(price_pct_of_notional) AS price_pct_of_notional,
       AVG(CAST(sensitivities_json->>'delta' AS DOUBLE)) AS avg_delta
FROM riskcube_cells
WHERE version_id = <returned version_id>
  AND scenario_id IN (<base scenario id>, <spot-down scenario id>, <vol-up scenario id>)
GROUP BY scenario_id, scenario_key, instrument_id
ORDER BY instrument_id, scenario_id;
```

Inspect scenario changes with a window function:

```sql
WITH instrument_scenarios AS (
  SELECT scenario_id, scenario_key, instrument_id,
         SUM(pv_amount) AS pv_amount,
         AVG(CAST(sensitivities_json->>'delta' AS DOUBLE)) AS delta
  FROM riskcube_cells
  WHERE version_id = <returned version_id>
  GROUP BY scenario_id, scenario_key, instrument_id
)
SELECT *,
       pv_amount - LAG(pv_amount) OVER (
         PARTITION BY instrument_id ORDER BY scenario_id
       ) AS pv_change_vs_previous_scenario
FROM instrument_scenarios
ORDER BY instrument_id, scenario_id;
```

## 7. Verify remote Parquet objects

Call `gcs_read_parquet` on one returned object name, or use an OLAP query over the configured Parquet dataset. Confirm that the returned rows contain integer `version_id` and `scenario_id`, stable `version_key` and `scenario_key`, `instance_id`, valuation fields, sensitivity JSON, and explainability JSON.

A successful manual promotion is complete only when the catalog rows, instance rows, Parquet objects, and OLAP results agree on the same version/scenario/instance mapping.
