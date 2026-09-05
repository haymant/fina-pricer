from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from .core import PricingRequest, sensitivity
from .gcs import gcs_status, load_local_env, read_parquet_from_gcs
from .scenario_builder import ScenarioBuilder
from .storage import RiskCubeStore, execute_scenario_batch

load_local_env()
_store = RiskCubeStore(
    os.getenv("RISKCUBE_DUCKDB_PATH", ":memory:"),
    os.getenv("RISKCUBE_PARQUET_ROOT", "data/riskcube"),
)

allowed_hosts = [
    host.strip()
    for host in os.getenv(
        "ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],localhost:*,127.0.0.1:*,[::1]:*,fina-pricer.vercel.app,fina-pricer.vercel.app:*"
    ).split(",")
    if host.strip()
]
mcp = FastMCP(
    "riskcube-pricing",
    stateless_http=True,
    transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Any) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "riskcube-pricing", "gcs": gcs_status()})


app = mcp.streamable_http_app()


@mcp.tool()
def pricing_and_sensitivity(request: dict[str, Any]) -> dict[str, Any]:
    """Price a structured product and return an explainable RiskCube."""
    parsed = PricingRequest.model_validate(request)
    return sensitivity(parsed)


@mcp.tool()
def scenario_create(scenario: dict[str, Any]) -> dict[str, Any]:
    """Create or replace a scenario definition in the scenario catalog."""
    built = ScenarioBuilder(
        scenario.get("scenario_name", scenario.get("scenario_id", "scenario")),
        scenario.get("base_market_data_datetime", "1970-01-01T00:00:00Z"),
        scenario.get("trade_repository_snapshot_datetime", "1970-01-01T00:00:00Z"),
        scenario_id=scenario.get("scenario_id"),
        scenario_version=str(scenario.get("scenario_version", "1")),
        materialization_mode=scenario.get("materialization_mode", "rules"),
    )
    built.scenario.update(scenario)
    payload = built.build()
    _store.register_scenario(payload)
    return payload


@mcp.tool()
def scenario_get(scenario_id: str) -> dict[str, Any]:
    """Retrieve one scenario definition without exposing storage credentials."""
    result = _store.get_scenario(scenario_id)
    if result is None:
        raise ValueError(f"scenario not found: {scenario_id}")
    return result


@mcp.tool()
def scenario_list() -> list[dict[str, Any]]:
    """List scenario definitions ordered by catalog creation time."""
    return _store.list_scenarios()


@mcp.tool()
def scenario_update(scenario: dict[str, Any]) -> dict[str, Any]:
    """Update a scenario by replacing its catalog definition."""
    if not scenario.get("scenario_id"):
        raise ValueError("scenario_id is required for update")
    return scenario_create(scenario)


@mcp.tool()
def scenario_delete(scenario_id: str) -> dict[str, Any]:
    """Delete a scenario definition when no foreign-key dependency blocks it."""
    return {"scenario_id": scenario_id, "deleted": _store.delete_scenario(scenario_id)}


@mcp.tool()
def scenario_trigger(
    scenario_id: str,
    requests: list[dict[str, Any]],
    batch_id: str | None = None,
    version: str = "1",
) -> dict[str, Any]:
    """Materialize and price a scenario batch, persisting a versioned RiskCube partition."""
    scenario = _store.get_scenario(scenario_id)
    if scenario is None:
        raise ValueError(f"scenario not found: {scenario_id}")
    request_pairs = [(str(item.get("case_id", index)), item["request"]) for index, item in enumerate(requests)]
    return execute_scenario_batch(request_pairs, scenario, _store, batch_id=batch_id, version=version)


@mcp.tool()
def olap_query(sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
    """Run a read-only DuckDB OLAP query over riskcube_cells."""
    normalized = sql.strip().lower()
    if not normalized.startswith(("select", "with")):
        raise ValueError("olap_query accepts only SELECT or WITH queries")
    forbidden = ("insert ", "update ", "delete ", "drop ", "create ", "alter ", "copy ", "install ", "load ")
    if any(token in normalized for token in forbidden) or ";" in normalized.rstrip(";"):
        raise ValueError("olap_query is read-only")
    result = _store.connection.execute(sql, parameters or []).fetchall()
    columns = [item[0] for item in _store.connection.description]
    return {"columns": columns, "rows": [list(row) for row in result], "row_count": len(result)}


@mcp.tool()
def gcs_read_parquet(object_name: str) -> dict[str, Any]:
    """Read a Parquet object from the configured GCS bucket using DuckDB S3 interoperability."""
    rows = read_parquet_from_gcs(_store.connection, object_name)
    columns = [item[0] for item in _store.connection.description]
    return {"columns": columns, "rows": [list(row) for row in rows], "row_count": len(rows)}


@mcp.tool()
def gcs_configuration_status() -> dict[str, Any]:
    """Return non-secret GCS configuration status; credential values are never returned."""
    return gcs_status()


@mcp.prompt()
def fina_scenario_guidance() -> str:
    return "Use scenario_create/list/get/update/delete for catalog management, then scenario_trigger to materialize a versioned RiskCube partition. Use VIRTUAL_CURRENT_REPORT for a current-market report template."


@mcp.prompt()
def fina_olap_guidance() -> str:
    return "Use olap_query with SELECT/WITH over riskcube_cells. Prefer grouped, pivoted, ROLLUP, and window-function queries over scalar extraction; use scenario, version, and coordinates as dimensions."


@mcp.prompt()
def fina_gcs_guidance() -> str:
    return "Use gcs_configuration_status for masked diagnostics and gcs_read_parquet for configured-bucket reads. Never print, return, or persist S3_API_KEY or S3_API_SECRET."


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
