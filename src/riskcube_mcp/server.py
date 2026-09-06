from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from .core import PricingRequest, sensitivity
from .gcs import gcs_status, load_local_env, read_parquet_from_gcs
from .scenario_builder import ScenarioBuilder
from .storage import (
    STORAGE_MODES,
    RiskCubeStore,
    _drop_view_if_present,
    execute_scenario_batch,
)

load_local_env()


def _runtime_root() -> str:
    return (os.getenv("RISKCUBE_PARQUET_ROOT") or os.getenv("S3_BUCKET_NAME") or "").strip() or (
        "/tmp/riskcube" if os.getenv("VERCEL") else "data/riskcube"
    )


def _resolve_storage_mode(root: str) -> str:
    """Return the effective persistence mode: env override wins, remote uris and Vercel default to s3."""
    env_mode = os.getenv("RISKCUBE_STORAGE_MODE")
    if env_mode in STORAGE_MODES:
        return env_mode
    bucket_or_uri = (os.getenv("RISKCUBE_PARQUET_ROOT") or os.getenv("S3_BUCKET_NAME") or "").strip()
    if bucket_or_uri.startswith(("s3://", "gs://")) or (bucket_or_uri and os.getenv("VERCEL")):
        return "s3"
    return "local"


_parquet_root = _runtime_root()
_default_storage_mode = _resolve_storage_mode(_parquet_root)
_store = RiskCubeStore(os.getenv("RISKCUBE_DUCKDB_PATH", ":memory:"), _parquet_root, storage_mode=_default_storage_mode)

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
    return JSONResponse({"status": "ok", "service": "riskcube-pricing", "gcs": gcs_status(), "storage_mode": _store.mode})


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
    numeric_id = _store.register_scenario(payload)
    stored = _store.get_scenario(numeric_id)
    if stored is None:
        raise RuntimeError("scenario was registered but could not be read back")
    return {**payload, "scenario_id": stored["scenario_id"], "scenario_key": stored["scenario_key"]}


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
def scenario_delete(scenario_id: str | int) -> dict[str, Any]:
    """Delete a scenario definition only when no materialized instance references it."""
    scenario = _store.get_scenario(scenario_id)
    if scenario is None:
        return {"scenario_id": scenario_id, "deleted": False}
    references = _store.connection.execute("SELECT count(*) FROM riskcube_instances WHERE scenario_id = ?", [scenario["scenario_id"]]).fetchone()[0]
    if references:
        raise ValueError(f"scenario {scenario_id} has {references} materialized instance(s); create a new scenario key instead")
    return {"scenario_id": scenario["scenario_id"], "scenario_key": scenario["scenario_key"], "deleted": _store.delete_scenario(scenario["scenario_id"])}


@mcp.tool()
def version_list() -> list[dict[str, Any]]:
    """List durable version catalog entries and their integer IDs."""
    return _store.list_versions()


@mcp.tool()
def scenario_trigger(
    scenario_id: str | int,
    requests: list[dict[str, Any]],
    batch_id: str | None = None,
    version: str | int = "1",
) -> dict[str, Any]:
    """Materialize and price one scenario batch.

    Requests is a list of {case_id, request} pricing batches; use one entry per
    instrument in the target slice/selection. batch_id groups several instances
    across slices or reruns into one logical run so cubes stay correlated;
    when omitted a timestamp-based batch id is generated. Each call produces a
    fresh instance whose cells are persisted as a version/scenario partition.
    """
    scenario = _store.get_scenario(scenario_id)
    if scenario is None:
        raise ValueError(f"scenario not found: {scenario_id}")
    request_pairs = [(str(item.get("case_id", index)), item["request"]) for index, item in enumerate(requests)]
    return execute_scenario_batch(request_pairs, scenario, _store, batch_id=batch_id, version=version)


@mcp.tool()
def olap_query(
    sql: str,
    parameters: list[Any] | None = None,
    version_id: int | None = None,
    scenario_id: int | None = None,
    version_key: str | None = None,
    scenario_key: str | None = None,
) -> dict[str, Any]:
    """Run a read-only DuckDB OLAP query over riskcube_cells.

    By default queries the in-memory riskcube_cells catalog. Pass version_id /
    scenario_id (or version_key / scenario_key) to query a persisted partition
    directly from the Parquet store — this works after a server restart
    without re-materialising anything. Both version and scenario must be given
    together.
    """
    normalized = sql.strip().lower()
    if not normalized.startswith(("select", "with")):
        raise ValueError("olap_query accepts only SELECT or WITH queries")
    forbidden = ("insert ", "update ", "delete ", "drop ", "create ", "alter ", "copy ", "install ", "load ")
    if any(token in normalized for token in forbidden) or ";" in normalized.rstrip(";"):
        raise ValueError("olap_query is read-only")

    scoped = any(value is not None for value in (version_id, scenario_id, version_key, scenario_key))
    if not scoped:
        result = _store.connection.execute(sql, parameters or []).fetchall()
        columns = [item[0] for item in _store.connection.description]
        return {"columns": columns, "rows": [list(row) for row in result], "row_count": len(result)}

    if version_id is None and version_key is not None:
        version_id = _store.resolve_version_id(version_key)
    if scenario_id is None and scenario_key is not None:
        scenario_id = _store.resolve_scenario_id(scenario_key)
    if version_id is None or scenario_id is None:
        raise ValueError("partition-scoped olap_query requires both a version and a scenario (ids or keys)")

    glob_pattern = _store.partition_glob(int(version_id), int(scenario_id))
    quoted_glob = glob_pattern.replace("'", "''")
    _drop_view_if_present(_store.connection, "riskcube_cells")
    _store.connection.execute(
        f"CREATE OR REPLACE TEMP VIEW riskcube_cells AS SELECT * FROM read_parquet('{quoted_glob}')",
    )
    try:
        result = _store.connection.execute(sql, parameters or []).fetchall()
        columns = [item[0] for item in _store.connection.description]
        return {"columns": columns, "rows": [list(row) for row in result], "row_count": len(result)}
    finally:
        _drop_view_if_present(_store.connection, "riskcube_cells")


@mcp.tool()
def riskcube_partitions() -> list[dict[str, Any]]:
    """List persisted RiskCube partitions (distinct version × scenario snapshots).

    Derived from the hive-partitioned Parquet store so it stays correct after a
    warm-instance restart. Each entry carries version_id/version_key,
    scenario_id/scenario_key, cell_count, instance_count, first/last run times
    and whether it was read from parquet or the in-memory catalog.
    """
    return _store.list_partitions()


@mcp.tool()
def slice_create(definition: dict[str, Any]) -> dict[str, Any]:
    """Create or replace an instrument slice definition (filter over instruments)."""
    stored = _store.register_slice(definition)
    if stored is None:
        raise RuntimeError("slice was registered but could not be read back")
    return stored


@mcp.tool()
def slice_update(definition: dict[str, Any]) -> dict[str, Any]:
    """Update an instrument slice definition; slice_id or slice_key identifies it."""
    if not definition.get("slice_id") and not definition.get("slice_key"):
        raise ValueError("slice_id or slice_key is required for update")
    stored = _store.register_slice(definition)
    if stored is None:
        raise RuntimeError("slice was updated but could not be read back")
    return stored


@mcp.tool()
def slice_get(slice_id: str | int) -> dict[str, Any]:
    """Retrieve one instrument slice definition."""
    result = _store.get_slice(slice_id)
    if result is None:
        raise ValueError(f"slice not found: {slice_id}")
    return result


@mcp.tool()
def slice_list() -> list[dict[str, Any]]:
    """List instrument slice definitions ordered newest first."""
    return _store.list_slices()


@mcp.tool()
def slice_delete(slice_id: str | int) -> dict[str, Any]:
    """Delete an instrument slice definition."""
    return {"slice_id": slice_id, "deleted": _store.delete_slice(slice_id)}


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


@mcp.tool()
def storage_status() -> dict[str, Any]:
    """Return the current cube persistence mode and non-secret storage diagnostics.

    mode is one of: s3 (default, persists cells + catalogs as Parquet in the
    configured bucket), memory (keeps everything in the DuckDB catalog), or
    local (dev-only filesystem root).
    """
    return {**_store.status(), "default_mode": _default_storage_mode, "gcs": gcs_status()}


@mcp.tool()
def set_storage_mode(mode: str) -> dict[str, Any]:
    """Switch cube persistence to 's3' (Parquet in the configured bucket), 'memory', or 'local'.

    Cells already materialized in the in-memory catalog remain queryable; only
    the persistence target for new partitions is changed.
    """
    previous = _store.mode
    status = _store.set_mode(mode)
    return {**status, "previous_mode": previous, "gcs": gcs_status()}


@mcp.prompt()
def fina_scenario_guidance() -> str:
    return "Persist each scenario with scenario_create and verify scenario_id (integer surrogate) plus scenario_key (stable business key) using scenario_get/list. Trigger a re-price with scenario_trigger, passing one {case_id, request} per target instrument; group instances across instruments and reruns with a shared batch_id, then retain instance_id and partition paths. Versions identify report-snapshot configurations and are auto-registered on first use. Define reusable instrument subsets as slices with slice_create (id, name, conditions over fields like isin/name/symbol/product_type/notional/currency) and list them with slice_list."


@mcp.prompt()
def fina_olap_guidance() -> str:
    return "Use olap_query with SELECT/WITH over riskcube_cells. Filter first on integer version_id and scenario_id, then use version_key and scenario_key for display. Prefer grouped, pivoted, ROLLUP, and window-function queries over scalar extraction; use instance_id for immutable run tracing. To inspect a persisted partition after a restart, pass version_id + scenario_id (or their keys) to olap_query — it then reads the matching Parquet partition directly. Use riskcube_partitions to list which version×scenario partitions exist."


@mcp.prompt()
def fina_gcs_guidance() -> str:
    return "Use gcs_configuration_status for masked diagnostics and gcs_read_parquet for configured-bucket reads. Use storage_status for the current persistence mode and set_storage_mode to toggle between 's3' (default) and 'memory'. Never print, return, or persist S3_API_KEY or S3_API_SECRET."


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
