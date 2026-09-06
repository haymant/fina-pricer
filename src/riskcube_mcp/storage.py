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
from .gcs import GCSConfigurationError, configure_duckdb_gcs
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

STORAGE_MODES = ("memory", "local", "s3")

SLICE_FIELDS = (
    "instrument_id",
    "isin",
    "name",
    "symbol",
    "product_type",
    "family",
    "group",
    "strategy_id",
    "portfolio",
    "currency",
    "notional",
    "status",
    "indicative",
)

SLICE_OPS = ("eq", "neq", "contains", "starts_with", "in", "not_in", "gt", "gte", "lt", "lte", "regex")

SLICE_PARTITION_GLOB = "version_id=*/scenario_id=*/*.parquet"


class RiskCubeStore:
    """In-memory DuckDB catalog with append-only Parquet RiskCube partitions.

    ``mode`` selects where materialized RiskCube partitions and catalogs are
    persisted:

    - ``memory`` — keep cells/catalogs in the in-memory DuckDB catalog only; no
      Parquet files are written and catalogs are not restored from disk/bucket.
    - ``local``  — write Parquet partitions and catalogs under a local directory.
    - ``s3``     — write Parquet partitions and catalogs to an S3-compatible
      bucket (``s3://`` / ``gs://`` root via DuckDB httpfs, default mode when
      a remote root is configured).
    """

    def __init__(
        self,
        database: str = ":memory:",
        parquet_root: str | Path = "data/riskcube",
        storage_mode: str | None = None,
    ) -> None:
        self.connection = duckdb.connect(database)
        self._root_reference = str(parquet_root).rstrip("/")
        self.parquet_uri: str | None = None
        self.parquet_root: Path | None = None
        self.mode = storage_mode or self._derive_mode(self._root_reference)
        if self.mode not in STORAGE_MODES:
            raise ValueError(f"invalid storage mode {self.mode!r}; expected one of {', '.join(STORAGE_MODES)}")
        self._apply_storage_mode()
        self.initialize()
        if self.mode != "memory":
            self._restore_catalogs()

    @staticmethod
    def _derive_mode(root: str) -> str:
        return "s3" if root.startswith(("s3://", "gs://")) else "local"

    def _apply_storage_mode(self) -> None:
        if self.mode == "memory":
            self.parquet_uri = None
            self.parquet_root = None
        elif self.mode == "s3":
            self.parquet_uri = self._root_reference
            self.parquet_root = None
        else:
            self.parquet_uri = None
            candidate = Path(self._root_reference)
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                candidate = Path("/tmp/riskcube")
                candidate.mkdir(parents=True, exist_ok=True)
            self.parquet_root = candidate

    def set_mode(self, mode: str) -> dict[str, Any]:
        """Switch persistence between memory/local/s3 on the same catalog."""
        if mode not in STORAGE_MODES:
            raise ValueError(f"invalid storage mode {mode!r}; expected one of {', '.join(STORAGE_MODES)}")
        if mode == self.mode:
            return self.status()
        if mode == "s3" and not self._root_reference.startswith(("s3://", "gs://")):
            raise GCSConfigurationError(
                "S3 storage requires an s3:// or gs:// parquet root (configure RISKCUBE_PARQUET_ROOT or S3_BUCKET_NAME)"
            )
        self.mode = mode
        self._apply_storage_mode()
        return self.status()

    def status(self) -> dict[str, Any]:
        """Return the current mode and non-secret persistence details."""
        return {
            "mode": self.mode,
            "modes": list(STORAGE_MODES),
            "parquet": {
                "uri": self.parquet_uri,
                "root": str(self.parquet_root) if self.parquet_root else None,
                "catalog": self._catalog_object("versions") if self.mode != "memory" else None,
                "partition_pattern": (
                    f"{self.parquet_uri}/version_id=*/scenario_id=*/*.parquet" if self.parquet_uri else None
                ),
            },
        }

    def initialize(self) -> None:
        self.connection.sql("CREATE SEQUENCE IF NOT EXISTS version_id_seq START 1")
        self.connection.sql("CREATE SEQUENCE IF NOT EXISTS scenario_id_seq START 1")
        self.connection.sql("CREATE SEQUENCE IF NOT EXISTS slice_id_seq START 1")
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS version_catalog (
                version_id BIGINT PRIMARY KEY,
                version_key VARCHAR NOT NULL UNIQUE,
                version_name VARCHAR NOT NULL,
                metadata_json JSON,
                created_at TIMESTAMP NOT NULL
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS scenario_catalog (
                scenario_id BIGINT PRIMARY KEY,
                scenario_key VARCHAR NOT NULL UNIQUE,
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
                version_id BIGINT NOT NULL,
                version_key VARCHAR NOT NULL,
                scenario_id BIGINT NOT NULL,
                scenario_key VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                requested_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                instrument_count BIGINT DEFAULT 0,
                cell_count BIGINT DEFAULT 0,
                partition_count BIGINT DEFAULT 0
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS riskcube_partitions (
                partition_id VARCHAR PRIMARY KEY,
                instance_id VARCHAR NOT NULL,
                version_id BIGINT NOT NULL,
                scenario_id BIGINT NOT NULL,
                parquet_path VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS riskcube_cells (
                instance_id VARCHAR NOT NULL,
                batch_id VARCHAR NOT NULL,
                version_id BIGINT NOT NULL,
                scenario_id BIGINT NOT NULL,
                version_key VARCHAR NOT NULL,
                scenario_key VARCHAR NOT NULL,
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
        self.connection.sql("""
            CREATE TABLE IF NOT EXISTS slice_catalog (
                slice_id BIGINT PRIMARY KEY,
                slice_key VARCHAR NOT NULL UNIQUE,
                slice_name VARCHAR NOT NULL,
                slice_kind VARCHAR NOT NULL,
                match_any BOOLEAN NOT NULL DEFAULT false,
                conditions_json JSON,
                metadata_json JSON,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

    def register_version(self, version: str | int, *, name: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        version_key = str(version)
        row = self.connection.execute("SELECT version_id, version_key, version_name FROM version_catalog WHERE version_key = ?", [version_key]).fetchone()
        if row is None:
            version_id = int(self.connection.execute("SELECT nextval('version_id_seq')").fetchone()[0])
            self.connection.execute(
                "INSERT INTO version_catalog VALUES (?, ?, ?, ?, ?)",
                [version_id, version_key, name or version_key, _json(metadata or {}), _now()],
            )
            self._persist_catalogs()
            return {"version_id": version_id, "version_key": version_key, "version_name": name or version_key}
        return {"version_id": int(row[0]), "version_key": str(row[1]), "version_name": str(row[2])}

    def register_scenario(self, scenario: dict[str, Any]) -> int:
        scenario_key = str(scenario.get("scenario_key", scenario["scenario_id"]))
        existing = self.connection.execute("SELECT scenario_id FROM scenario_catalog WHERE scenario_key = ?", [scenario_key]).fetchone()
        scenario_id = int(existing[0]) if existing else int(self.connection.execute("SELECT nextval('scenario_id_seq')").fetchone()[0])
        self.connection.execute("DELETE FROM scenario_catalog WHERE scenario_id = ?", [scenario_id])
        self.connection.execute(
            """
            INSERT INTO scenario_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                scenario_id,
                scenario_key,
                str(scenario.get("scenario_version", "1")),
                str(scenario.get("scenario_name", scenario_key)),
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
        self._persist_catalogs()
        return scenario_id

    def create_instance(self, batch_id: str, version: str | int, scenario_id: str | int, instrument_count: int = 0) -> str:
        version_meta = self.register_version(version)
        scenario_row = self.connection.execute("SELECT scenario_id, scenario_key FROM scenario_catalog WHERE scenario_id = ? OR scenario_key = ?", [scenario_id if str(scenario_id).isdigit() else -1, str(scenario_id)]).fetchone()
        if scenario_row is None:
            raise KeyError(f"scenario not found: {scenario_id}")
        scenario_numeric_id, scenario_key = int(scenario_row[0]), str(scenario_row[1])
        instance_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO riskcube_instances (instance_id, batch_id, version_id, version_key, scenario_id, scenario_key, status, requested_at, instrument_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [instance_id, batch_id, version_meta["version_id"], version_meta["version_key"], scenario_numeric_id, scenario_key, "RUNNING", _now(), instrument_count],
        )
        return instance_id

    def write_result(
        self,
        instance_id: str,
        batch_id: str,
        version: str | int,
        scenario_id: str | int,
        result: dict[str, Any],
        *,
        case_id: str | None = None,
        instrument_id: str | None = None,
    ) -> int:
        valuation = result.get("RiskCube", {}).get("valuation", {})
        explainability = result.get("explainability", {})
        instance_meta = self.connection.execute("SELECT version_id, version_key, scenario_id, scenario_key FROM riskcube_instances WHERE instance_id = ?", [instance_id]).fetchone()
        if instance_meta is None:
            raise KeyError(instance_id)
        version_id, version_key, scenario_numeric_id, scenario_key = instance_meta
        rows: list[list[Any]] = []
        for cell in result.get("RiskCube", {}).get("cells", []):
            rfk = cell.get("rfk", {})
            coordinates = cell.get("coordinates", {key: rfk[key] for key in AXIS_COLUMNS if key in rfk})
            rows.append([
                instance_id, batch_id, version_id, scenario_numeric_id, version_key, scenario_key, case_id, instrument_id,
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
        row = self.connection.execute("SELECT batch_id, version_id, scenario_id FROM riskcube_instances WHERE instance_id = ?", [instance_id]).fetchone()
        if row is None:
            raise KeyError(instance_id)
        _batch_id, version_id, scenario_id = row
        count_row = self.connection.execute("SELECT count(*) FROM riskcube_cells WHERE instance_id = ?", [instance_id]).fetchone()
        count = int(count_row[0]) if count_row is not None else 0
        if self.mode == "memory":
            self.connection.execute(
                "UPDATE riskcube_instances SET status = ?, completed_at = ?, cell_count = ?, partition_count = 0 WHERE instance_id = ?",
                [status, _now(), count, instance_id],
            )
            return []
        relative_path = f"version_id={int(version_id)}/scenario_id={int(scenario_id)}/instance_id={_safe(instance_id)}.parquet"
        if self.parquet_uri is not None:
            configure_duckdb_gcs(self.connection)
            path = f"{self.parquet_uri}/{relative_path}"
        else:
            if self.parquet_root is None:
                raise RuntimeError("local parquet root is not configured")
            partition_dir = self.parquet_root / f"version_id={int(version_id)}" / f"scenario_id={int(scenario_id)}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            path = str(partition_dir / f"instance_id={_safe(instance_id)}.parquet")
        sql_path = path.replace("'", "''")
        self.connection.execute(
            f"COPY (SELECT * FROM riskcube_cells WHERE instance_id = ?) TO '{sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            [instance_id],
        )
        partition_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO riskcube_partitions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [partition_id, instance_id, int(version_id), int(scenario_id), str(path), count, _now()],
        )
        self.connection.execute(
            "UPDATE riskcube_instances SET status = ?, completed_at = ?, cell_count = ?, partition_count = 1 WHERE instance_id = ?",
            [status, _now(), count, instance_id],
        )
        return [str(path)]

    def query_cells(self, sql: str = "SELECT * FROM riskcube_cells", parameters: Iterable[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.connection.execute(sql, list(parameters or [])).fetchall()

    def query_parquet(self, version: str | int | None = None, scenario_id: str | int | None = None) -> list[tuple[Any, ...]]:
        if self.mode == "memory":
            return []
        if self.parquet_uri is not None:
            configure_duckdb_gcs(self.connection)
            pattern = f"{self.parquet_uri}/version_id=*/scenario_id=*/*.parquet"
        else:
            if self.parquet_root is None:
                return []
            pattern = str(self.parquet_root / "version_id=*" / "scenario_id=*" / "*.parquet")
        clauses: list[str] = []
        params: list[Any] = []
        if version is not None:
            clauses.append("version_id = ?")
            params.append(int(version))
        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(int(scenario_id))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return self.connection.execute("SELECT * FROM read_parquet(?)" + where, [pattern, *params]).fetchall()

    def get_scenario(self, scenario_id: str | int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT scenario_id, scenario_key, scenario_version, scenario_name, scenario_kind, materialization_mode, base_market_data_datetime, trade_repository_snapshot_datetime, rules_json, market_data_snapshot_json, request_template_json, metadata_json FROM scenario_catalog WHERE scenario_id = ? OR scenario_key = ?",
            [scenario_id if str(scenario_id).isdigit() else -1, str(scenario_id)],
        ).fetchone()
        if row is None:
            return None
        keys = ["scenario_id", "scenario_key", "scenario_version", "scenario_name", "scenario_kind", "materialization_mode", "base_market_data_datetime", "trade_repository_snapshot_datetime", "market_data_manipulations", "market_data_snapshot", "request_template", "metadata"]
        result = dict(zip(keys, row))
        for key in ("market_data_manipulations", "market_data_snapshot", "request_template", "metadata"):
            if isinstance(result[key], str):
                result[key] = json.loads(result[key])
        return result

    def list_scenarios(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT scenario_id FROM scenario_catalog ORDER BY created_at DESC").fetchall()
        scenarios: list[dict[str, Any]] = []
        for row in rows:
            scenario = self.get_scenario(str(row[0]))
            if scenario is not None:
                scenarios.append(scenario)
        return scenarios

    def delete_scenario(self, scenario_id: str | int) -> bool:
        existing = self.get_scenario(scenario_id)
        if existing is None:
            return False
        self.connection.execute("DELETE FROM scenario_catalog WHERE scenario_id = ?", [existing["scenario_id"]])
        return True

    def list_versions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT version_id, version_key, version_name, metadata_json, created_at FROM version_catalog ORDER BY version_id").fetchall()
        return [{"version_id": int(row[0]), "version_key": row[1], "version_name": row[2], "metadata": _parse_json(row[3]), "created_at": row[4]} for row in rows]

    def register_slice(self, definition: dict[str, Any]) -> dict[str, Any]:
        """Upsert a slice definition keyed by slice_key; returns the stored row."""
        slice_key = str(definition.get("slice_key") or definition.get("slice_name") or definition.get("slice_id") or "slice")
        existing = self.connection.execute("SELECT slice_id FROM slice_catalog WHERE slice_key = ?", [slice_key]).fetchone()
        slice_id = int(existing[0]) if existing else int(self.connection.execute("SELECT nextval('slice_id_seq')").fetchone()[0])
        conditions = definition.get("conditions") or []
        if not isinstance(conditions, list):
            raise TypeError("slice conditions must be a list of {field, op, value} objects")
        for condition in conditions:
            field = str(condition.get("field", ""))
            op = str(condition.get("op", ""))
            if field not in SLICE_FIELDS:
                raise ValueError(f"unsupported slice field {field!r}; expected one of {', '.join(SLICE_FIELDS)}")
            if op not in SLICE_OPS:
                raise ValueError(f"unsupported slice op {op!r}; expected one of {', '.join(SLICE_OPS)}")
        self.connection.execute("DELETE FROM slice_catalog WHERE slice_id = ?", [slice_id])
        self.connection.execute(
            """
            INSERT INTO slice_catalog (slice_id, slice_key, slice_name, slice_kind, match_any, conditions_json, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                slice_id,
                slice_key,
                str(definition.get("slice_name") or slice_key),
                str(definition.get("slice_kind", "filter")),
                bool(definition.get("match_any", False)),
                _json({"conditions": conditions}),
                _json({"description": definition.get("description"), "metadata": definition.get("metadata", {})}),
                _now(),
                _now(),
            ],
        )
        self._persist_catalogs()
        return self.get_slice(slice_id) or {"slice_id": slice_id}

    def get_slice(self, slice_id: str | int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT slice_id, slice_key, slice_name, slice_kind, match_any, conditions_json, metadata_json, updated_at FROM slice_catalog WHERE slice_id = ? OR slice_key = ?",
            [slice_id if str(slice_id).isdigit() else -1, str(slice_id)],
        ).fetchone()
        if row is None:
            return None
        conditions = _parse_json(row[5])
        metadata = _parse_json(row[6])
        return {
            "slice_id": int(row[0]),
            "slice_key": str(row[1]),
            "slice_name": str(row[2]),
            "slice_kind": str(row[3]),
            "match_any": bool(row[4]),
            "conditions": conditions.get("conditions", []) if isinstance(conditions, dict) else [],
            "description": (metadata or {}).get("description") if isinstance(metadata, dict) else None,
            "metadata": (metadata or {}).get("metadata", {}) if isinstance(metadata, dict) else {},
            "updated_at": row[7],
        }

    def list_slices(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT slice_id FROM slice_catalog ORDER BY created_at DESC, slice_id").fetchall()
        slices: list[dict[str, Any]] = []
        for row in rows:
            definition = self.get_slice(str(row[0]))
            if definition is not None:
                slices.append(definition)
        return slices

    def delete_slice(self, slice_id: str | int) -> bool:
        existing = self.get_slice(slice_id)
        if existing is None:
            return False
        self.connection.execute("DELETE FROM slice_catalog WHERE slice_id = ?", [existing["slice_id"]])
        self._persist_catalogs()
        return True

    def partition_glob(self, version_id: int | None = None, scenario_id: int | None = None) -> str:
        """Hive-partitioned glob over persisted instance parquets for a version/scenario partition."""
        if self.mode == "memory":
            raise RuntimeError("partition queries require persisted partitions (s3/local storage mode)")
        pattern = SLICE_PARTITION_GLOB
        if self.parquet_uri is not None:
            configure_duckdb_gcs(self.connection)
            root = f"{self.parquet_uri}/"
        elif self.parquet_root is not None:
            root = f"{self.parquet_root}/"
        else:
            raise RuntimeError("parquet storage is not configured")
        if version_id is not None:
            pattern = pattern.replace("version_id=*", f"version_id={int(version_id)}")
        if scenario_id is not None:
            pattern = pattern.replace("scenario_id=*", f"scenario_id={int(scenario_id)}")
        return f"{root}{pattern}"

    def resolve_version_id(self, version: str | int) -> int:
        row = self.connection.execute("SELECT version_id FROM version_catalog WHERE version_id = ? OR version_key = ?", [version if str(version).isdigit() else -1, str(version)]).fetchone()
        if row is None:
            raise KeyError(f"version not found: {version}")
        return int(row[0])

    def resolve_scenario_id(self, scenario_id: str | int) -> int:
        row = self.connection.execute("SELECT scenario_id FROM scenario_catalog WHERE scenario_id = ? OR scenario_key = ?", [scenario_id if str(scenario_id).isdigit() else -1, str(scenario_id)]).fetchone()
        if row is None:
            raise KeyError(f"scenario not found: {scenario_id}")
        return int(row[0])

    def list_partitions(self) -> list[dict[str, Any]]:
        """List distinct version/scenario partitions from the persisted parquet store.

        Survives warm-instance restarts: the inventory is derived from the
        hive-partitioned instance parquets rather than the in-memory catalog.
        In memory mode it falls back to the in-memory instance registry.
        """
        partitions: dict[tuple[int, int], dict[str, Any]] = {}
        rows: list[tuple[Any, ...]] = []
        if self.mode != "memory":
            try:
                glob = self.partition_glob()
                rows = self.connection.execute(
                    """
                    SELECT version_id, scenario_id, COUNT(*) AS cell_count,
                           COUNT(DISTINCT instance_id) AS instance_count,
                           MIN(created_at) AS first_at, MAX(created_at) AS last_at
                      FROM read_parquet(?, hive_partitioning = true)
                     GROUP BY 1, 2
                     ORDER BY 1 DESC, 2 DESC
                    """,
                    [glob],
                ).fetchall()
            except (duckdb.IOException, duckdb.CatalogException, duckdb.BinderException):
                rows = []
            for row in rows:
                version_id, scenario_id = int(row[0]), int(row[1])
                partitions[(version_id, scenario_id)] = {
                    "version_id": version_id,
                    "scenario_id": scenario_id,
                    "cell_count": int(row[2]),
                    "instance_count": int(row[3]),
                    "first_at": row[4],
                    "last_at": row[5],
                    "source": "parquet",
                }
        else:
            try:
                rows = self.connection.execute(
                    "SELECT version_id, scenario_id, SUM(cell_count), COUNT(*), MIN(created_at), MAX(created_at) FROM riskcube_instances GROUP BY 1, 2"
                ).fetchall()
            except duckdb.CatalogException:
                rows = []
            for row in rows:
                version_id, scenario_id = int(row[0]), int(row[1])
                partitions[(version_id, scenario_id)] = {
                    "version_id": version_id,
                    "scenario_id": scenario_id,
                    "cell_count": int(row[2]),
                    "instance_count": int(row[3]),
                    "first_at": row[4],
                    "last_at": row[5],
                    "source": "memory",
                }
        for (version_id, scenario_id), entry in partitions.items():
            version_key = self.connection.execute(
                "SELECT version_key FROM version_catalog WHERE version_id = ?", [version_id]
            ).fetchone()
            scenario_key = self.connection.execute(
                "SELECT scenario_key FROM scenario_catalog WHERE scenario_id = ?", [scenario_id]
            ).fetchone()
            entry["version_key"] = str(version_key[0]) if version_key else str(version_id)
            entry["scenario_key"] = str(scenario_key[0]) if scenario_key else str(scenario_id)
        return sorted(partitions.values(), key=lambda entry: (entry["last_at"] or ""), reverse=True)

    def _catalog_object(self, name: str) -> str:
        if self.mode == "memory":
            raise RuntimeError("catalog objects are not persisted in memory mode")
        if self.parquet_uri is not None:
            return f"{self.parquet_uri}/catalog/{name}.parquet"
        if self.parquet_root is None:
            raise RuntimeError("local parquet root is not configured")
        directory = self.parquet_root / "catalog"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"{name}.parquet")

    def _persist_catalogs(self) -> None:
        if self.mode == "memory":
            return
        if self.parquet_uri is not None:
            configure_duckdb_gcs(self.connection)
        for table, name in (("version_catalog", "versions"), ("scenario_catalog", "scenarios"), ("slice_catalog", "slices")):
            path = self._catalog_object(name).replace("'", "''")
            self.connection.execute(f"COPY (SELECT * FROM {table}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    def _restore_catalogs(self) -> None:
        if self.mode == "memory":
            return
        for table, name in (("version_catalog", "versions"), ("scenario_catalog", "scenarios"), ("slice_catalog", "slices")):
            try:
                if self.parquet_uri is not None:
                    configure_duckdb_gcs(self.connection)
                path = self._catalog_object(name)
                rows = self.connection.execute("SELECT * FROM read_parquet(?)", [path]).fetchall()
                if rows:
                    placeholders = ",".join(["?"] * len(rows[0]))
                    self.connection.execute(f"DELETE FROM {table}")
                    self.connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                    id_column = {
                        "version_catalog": "version_id",
                        "scenario_catalog": "scenario_id",
                        "slice_catalog": "slice_id",
                    }[table]
                    sequence = {
                        "version_catalog": "version_id_seq",
                        "scenario_catalog": "scenario_id_seq",
                        "slice_catalog": "slice_id_seq",
                    }[table]
                    max_id = self.connection.execute(f"SELECT max({id_column}) FROM {table}").fetchone()[0]
                    if max_id is not None:
                        next_id = int(self.connection.execute(f"SELECT nextval('{sequence}')").fetchone()[0])
                        while next_id <= int(max_id):
                            next_id = int(self.connection.execute(f"SELECT nextval('{sequence}')").fetchone()[0])
            except Exception:
                continue

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
    scenario_numeric_id = store.register_scenario(scenario)
    scenario_meta = store.get_scenario(scenario_numeric_id)
    if scenario_meta is None:
        raise KeyError(f"scenario not found after registration: {scenario_numeric_id}")
    batch = batch_id or f"batch-{int(time.time())}"
    request_list = list(requests)
    instance_id = store.create_instance(batch, version, scenario_numeric_id, len(request_list))
    cell_count = 0
    for case_id, request_payload in request_list:
        materialized = materialize_request(request_payload, scenario, market_data_resolver=market_data_resolver)
        request = PricingRequest.model_validate(materialized)
        result = sensitivity(request)
        cell_count += store.write_result(
            instance_id, batch, version, scenario_numeric_id, result,
            case_id=case_id,
            instrument_id=request.instrument.isin,
        )
    partitions = store.finalize_instance(instance_id)
    version_meta = store.register_version(version)
    return {
        "instance_id": instance_id,
        "batch_id": batch,
        "version_id": version_meta["version_id"],
        "version_key": version_meta["version_key"],
        "scenario_id": scenario_meta["scenario_id"],
        "scenario_key": scenario_meta["scenario_key"],
        "instrument_count": len(request_list),
        "cell_count": cell_count,
        "partitions": partitions,
    }


def _drop_view_if_present(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Drop a (possibly temp) view shadowing a same-named table; tables are left intact."""
    exists = connection.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [name]).fetchone()
    if not exists or int(exists[0]) == 0:
        return False
    connection.execute(f"DROP VIEW IF EXISTS {name}")
    return True


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"), default=str)


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
