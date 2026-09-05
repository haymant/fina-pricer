---
name: fina-olap
description: DuckDB OLAP over materialized RiskCube cells in memory or Parquet. Use for grouping, pivoting, rollups, window functions, scenario/version comparisons, AG Grid data services, and Tableau-style level-of-detail analysis.
---

# FInA OLAP

Use this skill for analytical queries over `riskcube_cells`, persisted RiskCube Parquet partitions, or scenario/version batches.

## Query route

Use the MCP `olap_query` tool for read-only SQL over the in-memory `riskcube_cells` table. Use `SELECT` or `WITH` only. Filter early by integer `scenario_id`, integer `version_id`, `instance_id`, or `instrument_id`; use `scenario_key` and `version_key` for display. For historical materializations, query Parquet with DuckDB and preserve the same scalar column contract.

The canonical dimensions are `type`, `underlying`, `currency_pair`, `expiry`, `strike`, `tenor`, `temporal_role`, `date`, and `surface_parameter`. Numeric sensitivities are stored in `sensitivities_json`; common fields such as `delta`, `gamma`, `vega`, `rho`, and `theta` should be projected into scalar columns when creating a serving view.

## Core patterns

Use `GROUP BY` for exposure summaries:

```sql
SELECT scenario_id, version_id, scenario_key, version_key, underlying, type,
       AVG(CAST(sensitivities_json->>'delta' AS DOUBLE)) AS avg_delta,
       SUM(pv_amount) AS pv_amount
FROM riskcube_cells
GROUP BY scenario_id, version_id, scenario_key, version_key, underlying, type;
```

Use `PIVOT` for AG Grid-style columns by risk-factor type. Use `ROLLUP` for totals by scenario, underlying, and type. Use window functions for scenario ranking, percentile bands, and before/after comparisons:

```sql
SELECT *,
       pv_amount - LAG(pv_amount) OVER (
         PARTITION BY instrument_id, underlying ORDER BY version_id
       ) AS pv_change
FROM riskcube_cells;
```

## Tableau-style LOD equivalents

Use a grouped subquery or window function instead of Tableau syntax. `{FIXED [Underlying] : SUM([Delta])}` becomes a grouped relation joined back to detail rows. `{INCLUDE [Scenario] : AVG([PV])}` becomes a grouping query at the requested grain. `{EXCLUDE [RiskFactorType] : SUM([Delta])}` becomes a window aggregate partitioned by all retained dimensions.

Always state the grain of the result. Distinguish cell-level totals from instrument-level totals, because each instrument can have multiple RiskCube cells.

## Performance

Prefer Parquet partition filters on integer `version_id` and `scenario_id`. Project only required columns. Aggregate before returning to an AG Grid client. Keep JSON fields for provenance and AI reconstruction, but use scalar axis and valuation columns for interactive filters. The canonical object layout is `version_id=<int>/scenario_id=<int>/instance_id=<uuid>.parquet`.
 Do not return millions of cells to the model when a grouped result is sufficient.

## Safety and provenance

`olap_query` is read-only. Do not use DDL, DML, `COPY`, `INSTALL`, or `LOAD` through the read-only query tool. Preserve integer `scenario_id`, integer `version_id`, stable `scenario_key`, stable `version_key`, `instance_id`, and `method` in analytical outputs so results can be traced to a pricing run.
