---
name: fina-scenario
description: Scenario catalog and scenario-driven RiskCube execution for structured-product risk. Use for creating, reading, updating, deleting, materializing, versioning, and triggering pricing scenarios through the RiskCube MCP server.
---

# FInA Scenario

Use this skill when a request involves scenario management, market-data shocks, current-market reports, scenario versions, or triggering a RiskCube batch.

## MCP workflow

1. Call `scenario_create` with a stable `scenario_key` supplied as the request's `scenario_id`, `scenario_version`, description, and either `market_data_snapshot` or `market_data_manipulations`. The catalog persists the definition and allocates an integer `scenario_id` surrogate.
2. Call `scenario_get` or `scenario_list` and retain both `scenario_id` and `scenario_key`. The integer is for joins and partition pruning; the key is for human governance and reproducibility.
3. Prepare `requests` as objects with `case_id` and a complete pricing request. Reuse the same request set when promoting multiple scenarios.
4. Call `scenario_trigger` once per scenario with the same `version`/`version_key`, a distinct `batch_id` or a common promotion batch, and the scenario's integer ID or stable key. The catalog allocates one integer `version_id` for the version key.
5. Preserve `instance_id`, `version_id`, `version_key`, `scenario_id`, `scenario_key`, `cell_count`, and Parquet partition paths for downstream OLAP.
6. Treat scenario definitions and version metadata as durable catalog records. A changed market snapshot or shock rule should create a new scenario key or version rather than silently changing a completed production run.

Use `scenario_update` to replace a definition deliberately before production use. Use `scenario_delete` only when no completed RiskCube instance depends on the scenario. Completed instances remain immutable and are addressed by their `instance_id` and integer partition IDs.

## Scenario forms

A materialized scenario carries `market_data_snapshot`. A rule scenario carries `market_data_manipulations` such as `spot_shift`, `vol_manipulation`, `fx_manipulation`, `ir_manipulation`, `dividend_manipulation`, and `fx_vol_manipulation`. Rules may include `underlying_name`, `currency_pair`, `curve_name`, and `filter_conditions`.

Use `ScenarioBuilder.current_report()` or `VIRTUAL_CURRENT_REPORT` for a report over all instruments at current evaluation time. A deployment should provide a market-data resolver for that virtual scenario; do not invent market data when no resolver is configured.

## Governance

Keep scenario keys and version keys immutable once used for a production batch. Integer `scenario_id` and `version_id` values are generated automatically and remain stable for the life of the catalog. Treat a changed shock rule or market snapshot as a new scenario key or version key. Record source timestamps, trade-repository snapshot timestamps, and scenario descriptions. Keep the complete request and materialized market snapshot available for reproducibility.

## Explainability

A scenario trigger returns a RiskCube instance, not only a scalar report. Preserve the mapping `scenario_id -> instance_id -> Parquet partition`. Query cells by scenario and version before comparing risk. For AI analysis, reconstruct nested JSON from flat rows using `case_id`, `instrument_id`, coordinates, valuation, and sensitivity JSON.

Never place credentials or access tokens in scenario definitions, request payloads, logs, or scenario metadata.
