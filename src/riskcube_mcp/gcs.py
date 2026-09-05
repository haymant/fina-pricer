"""GCS interoperability helpers for DuckDB's S3-compatible reader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb


def load_local_env(path: str | Path = ".env.local") -> None:
    """Load local key/value settings only when the variables are not already set."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class GCSConfigurationError(RuntimeError):
    """Raised when required GCS/S3 interoperability settings are missing."""


def gcs_configured() -> bool:
    return all(os.getenv(name) for name in ("S3_API_KEY", "S3_API_SECRET", "S3_BUCKET_NAME"))


def configure_duckdb_gcs(connection: duckdb.DuckDBPyConnection, *, secret_name: str = "fina_gcs") -> None:
    """Configure a DuckDB S3 secret from environment variables without logging them."""
    key_id = os.getenv("S3_API_KEY")
    secret = os.getenv("S3_API_SECRET")
    if not key_id or not secret:
        raise GCSConfigurationError("S3_API_KEY and S3_API_SECRET must be configured")
    endpoint = os.getenv("S3_ENDPOINT", "storage.googleapis.com").strip()
    if not endpoint:
        raise GCSConfigurationError("S3_ENDPOINT must not be empty")
    sql = (
        f"CREATE OR REPLACE SECRET {_identifier(secret_name)} ("
        f"TYPE S3, KEY_ID {_literal(key_id)}, SECRET {_literal(secret)}, "
        f"ENDPOINT {_literal(endpoint)}, URL_STYLE 'path', REGION 'auto')"
    )
    connection.execute(sql)


def configured_bucket() -> str:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise GCSConfigurationError("S3_BUCKET_NAME must be configured")
    return bucket.rstrip("/")


def gcs_object_uri(object_name: str) -> str:
    """Build an S3-compatible GCS URI from the configured bucket."""
    return f"{configured_bucket()}/{object_name.lstrip('/')}"


def read_parquet_from_gcs(
    connection: duckdb.DuckDBPyConnection,
    object_name: str,
    *,
    hive_partitioning: bool = True,
) -> list[tuple[Any, ...]]:
    configure_duckdb_gcs(connection)
    uri = gcs_object_uri(object_name)
    return connection.execute(
        "SELECT * FROM read_parquet(?, hive_partitioning = ?)",
        [uri, hive_partitioning],
    ).fetchall()


def gcs_status() -> dict[str, Any]:
    """Return non-secret configuration diagnostics suitable for health/reporting."""
    bucket = os.getenv("S3_BUCKET_NAME")
    return {
        "configured": gcs_configured(),
        "bucket": bucket,
        "endpoint": os.getenv("S3_ENDPOINT", "storage.googleapis.com"),
        "url_style": "path",
        "region": "auto",
        "credentials_present": bool(os.getenv("S3_API_KEY") and os.getenv("S3_API_SECRET")),
    }


def _identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError("secret_name must be alphanumeric or underscore")
    return value


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
