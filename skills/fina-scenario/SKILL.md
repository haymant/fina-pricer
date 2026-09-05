---
name: fina-scenario
description: Scenario catalog and scenario-driven RiskCube execution for structured-product risk. Use for creating, reading, updating, deleting, materializing, versioning, and triggering pricing scenarios through the RiskCube MCP server.
---

# FInA Scenario

Use this skill when a request involves scenario management, market-data shocks, current-market reports, scenario versions, or triggering a RiskCube batch.

## MCP workflow

1. Call `scenario_create` with a stable `scenario_id`, `scenario_version`, description, and either `market_data_snapshot` or `market_data_manipulations`.
2. Call `scenario_get` or `scenario_list` to verify catalog state.
3. Prepare `requests` as objects with `case_id` and a complete pricing request.
4. Call `scenario_trigger` with `scenario_id`, `version`, and optional `batch_id`.
5. Preserve the returned `instance_id`, `scenario_id`, `version`, `cell_count`, and Parquet partition paths for downstream OLAP.

Use `scenario_update` to replace a definition deliberately. Use `scenario_delete` only when no completed RiskCube instance depends on the scenario.

## Scenario forms

A materialized scenario carries `market_data_snapshot`. A rule scenario carries `market_data_manipulations` such as `spot_shift`, `vol_manipulation`, `fx_manipulation`, `ir_manipulation`, `dividend_manipulation`, and `fx_vol_manipulation`. Rules may include `underlying_name`, `currency_pair`, `curve_name`, and `filter_conditions`.

Use `ScenarioBuilder.current_report()` or `VIRTUAL_CURRENT_REPORT` for a report over all instruments at current evaluation time. A deployment should provide a market-data resolver for that virtual scenario; do not invent market data when no resolver is configured.

## Governance

Keep scenario IDs and versions immutable once used for a production batch. Treat a changed shock rule or market snapshot as a new version. Record source timestamps, trade-repository snapshot timestamps, and scenario descriptions. Keep the complete request and materialized market snapshot available for reproducibility.

## Explainability

A scenario trigger returns a RiskCube instance, not only a scalar report. Preserve the mapping `scenario_id -> instance_id -> Parquet partition`. Query cells by scenario and version before comparing risk. For AI analysis, reconstruct nested JSON from flat rows using `case_id`, `instrument_id`, coordinates, valuation, and sensitivity JSON.

Never place credentials or access tokens in scenario definitions, request payloads, logs, or scenario metadata.
