"""DuckDB orchestration and Parquet persistence for scenario-driven RiskCubes."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .core import PricingRequest, sensitivity
from .scenario_builder import materialize_request

AXIS_COLUMNS = [
    "type",
    "underlying",
    "currency_pair",
    "expiry",
    "strike",
    "tenor",
    "temporal_role",
    "date",
    "surface_parameter",
]


class RiskCubeStore:
    """In-memory DuckDB catalog with append-only Parquet RiskCube partitions."""

    def __init__(self, database: str = ":memory:", parquet_root: str | Path = "data/riskcube") -> None:
        self.connection = duckdb.connect(database)
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS scenario_catalog (
                scenario_id VARCHAR PRIMARY KEY,
                scenario_version VARCHAR NOT NULL,
                scenario_name VARCHAR NOT NULL,
                scenario_kind VARCHAR NOT NULL,
                materialization_mode VARCHAR NOT NULL,
                base_market_data_datetime TIMESTAMP,
                trade_repository_snapshot_datetime TIMESTAMP,
                rules_json JSON,
                market_data_snapshot_json JSON,
                request_template_json JSON,
                metadata_json JSON,
                created_at TIMESTAMP NOT NULL
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS riskcube_instances (
                instance_id VARCHAR PRIMARY KEY,
                batch_id VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                scenario_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                requested_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                instrument_count BIGINT DEFAULT 0,
                cell_count BIGINT DEFAULT 0,
                partition_count BIGINT DEFAULT 0,
                FOREIGN KEY (scenario_id) REFERENCES scenario_catalog(scenario_id)
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS riskcube_partitions (
                partition_id VARCHAR PRIMARY KEY,
                instance_id VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                scenario_id VARCHAR NOT NULL,
                parquet_path VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (instance_id) REFERENCES riskcube_instances(instance_id)
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS riskcube_cells (
                instance_id VARCHAR NOT NULL,
                batch_id VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                scenario_id VARCHAR NOT NULL,
                case_id VARCHAR,
                instrument_id VARCHAR,
                pv_amount DOUBLE,
                price_pct_of_notional DOUBLE,
                pv_currency VARCHAR,
                method VARCHAR,
                bump DOUBLE,
                type VARCHAR,
                underlying VARCHAR,
                currency_pair VARCHAR,
                expiry VARCHAR,
                strike DOUBLE,
                tenor VARCHAR,
                temporal_role VARCHAR,
                date VARCHAR,
                surface_parameter VARCHAR,
                sensitivities_json JSON,
                rfk_json JSON,
                coordinates_json JSON,
                explainability_json JSON,
                valuation_json JSON,
                created_at TIMESTAMP NOT NULL
            )
        """)

    def register_scenario(self, scenario: dict[str, Any]) -> str:
        scenario_id = str(scenario["scenario_id"])
        self.connection.execute(
            """
            INSERT OR REPLACE INTO scenario_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                scenario_id,
                str(scenario.get("scenario_version", "1")),
                str(scenario.get("scenario_name", scenario_id)),
                str(scenario.get("scenario_kind", "rule")),
                str(scenario.get("materialization_mode", "rules")),
                _timestamp(scenario.get("base_market_data_datetime")),
                _timestamp(scenario.get("trade_repository_snapshot_datetime")),
                _json(scenario.get("market_data_manipulations", [])),
                _json(scenario.get("market_data_snapshot")),
                _json(scenario.get("request_template")),
                _json({"description": scenario.get("description"), "metadata": scenario.get("metadata", {})}),
                _now(),
            ],
        )
        return scenario_id

    def create_instance(self, batch_id: str, version: str, scenario_id: str, instrument_count: int = 0) -> str:
        instance_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO riskcube_instances (instance_id, batch_id, version, scenario_id, status, requested_at, instrument_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [instance_id, batch_id, version, scenario_id, "RUNNING", _now(), instrument_count],
        )
        return instance_id

    def write_result(
        self,
        instance_id: str,
        batch_id: str,
        version: str,
        scenario_id: str,
        result: dict[str, Any],
        *,
        case_id: str | None = None,
        instrument_id: str | None = None,
    ) -> int:
        valuation = result.get("RiskCube", {}).get("valuation", {})
        explainability = result.get("explainability", {})
        rows: list[list[Any]] = []
        for cell in result.get("RiskCube", {}).get("cells", []):
            rfk = cell.get("rfk", {})
            coordinates = cell.get("coordinates", {key: rfk[key] for key in AXIS_COLUMNS if key in rfk})
            rows.append([
                instance_id, batch_id, version, scenario_id, case_id, instrument_id,
                cell.get("pv_amount", result.get("PV_amount", result.get("PV"))),
                cell.get("price_pct_of_notional", result.get("price_pct_of_notional")),
                result.get("PV_currency", valuation.get("pv_currency")),
                cell.get("method"), cell.get("bump"),
                coordinates.get("type"), coordinates.get("underlying"), coordinates.get("currency_pair"),
                coordinates.get("expiry"), coordinates.get("strike"), coordinates.get("tenor"),
                coordinates.get("temporal_role"), coordinates.get("date"), coordinates.get("surface_parameter"),
                _json(cell.get("sensitivities", {})), _json(rfk), _json(coordinates),
                _json(explainability), _json(valuation), _now(),
            ])
        if rows:
            self.connection.executemany(
                "INSERT INTO riskcube_cells VALUES (" + ",".join(["?"] * len(rows[0])) + ")",
                rows,
            )
        return len(rows)

    def finalize_instance(self, instance_id: str, *, status: str = "COMPLETED") -> list[str]:
        row = self.connection.execute("SELECT batch_id, version, scenario_id FROM riskcube_instances WHERE instance_id = ?", [instance_id]).fetchone()
        if row is None:
            raise KeyError(instance_id)
        _batch_id, version, scenario_id = row
        partition_dir = self.parquet_root / f"version={_safe(version)}" / f"scenario_id={_safe(scenario_id)}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / f"instance_id={_safe(instance_id)}.parquet"
        sql_path = str(path).replace("'", "''")
        self.connection.execute(
            f"COPY (SELECT * FROM riskcube_cells WHERE instance_id = ?) TO '{sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            [instance_id],
        )
        count_row = self.connection.execute("SELECT count(*) FROM riskcube_cells WHERE instance_id = ?", [instance_id]).fetchone()
        count = int(count_row[0]) if count_row is not None else 0
        partition_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO riskcube_partitions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [partition_id, instance_id, version, scenario_id, str(path), count, _now()],
        )
        self.connection.execute(
            "UPDATE riskcube_instances SET status = ?, completed_at = ?, cell_count = ?, partition_count = 1 WHERE instance_id = ?",
            [status, _now(), count, instance_id],
        )
        return [str(path)]

    def query_cells(self, sql: str = "SELECT * FROM riskcube_cells", parameters: Iterable[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.connection.execute(sql, list(parameters or [])).fetchall()

    def query_parquet(self, version: str | None = None, scenario_id: str | None = None) -> list[tuple[Any, ...]]:
        pattern = str(self.parquet_root / "version=*" / "scenario_id=*" / "*.parquet")
        clauses: list[str] = []
        params: list[Any] = []
        if version:
            clauses.append("version = ?")
            params.append(version)
        if scenario_id:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return self.connection.execute("SELECT * FROM read_parquet(?)" + where, [pattern, *params]).fetchall()

    def close(self) -> None:
        self.connection.close()


def execute_scenario_batch(
    requests: Iterable[tuple[str, dict[str, Any]]],
    scenario: dict[str, Any],
    store: RiskCubeStore,
    *,
    batch_id: str | None = None,
    version: str = "1",
    market_data_resolver: Any | None = None,
) -> dict[str, Any]:
    """Price requests for one scenario and persist the flattened cells."""
    scenario_id = store.register_scenario(scenario)
    batch = batch_id or f"batch-{int(time.time())}"
    request_list = list(requests)
    instance_id = store.create_instance(batch, version, scenario_id, len(request_list))
    cell_count = 0
    for case_id, request_payload in request_list:
        materialized = materialize_request(request_payload, scenario, market_data_resolver=market_data_resolver)
        request = PricingRequest.model_validate(materialized)
        result = sensitivity(request)
        cell_count += store.write_result(
            instance_id, batch, version, scenario_id, result,
            case_id=case_id,
            instrument_id=request.instrument.isin,
        )
    partitions = store.finalize_instance(instance_id)
    return {"instance_id": instance_id, "batch_id": batch, "scenario_id": scenario_id, "version": version, "instrument_count": len(request_list), "cell_count": cell_count, "partitions": partitions}


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"), default=str)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
