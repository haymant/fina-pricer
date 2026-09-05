from __future__ import annotations

import json
from pathlib import Path

from riskcube_mcp.scenario_builder import ScenarioBuilder, materialize_request
from riskcube_mcp.storage import RiskCubeStore, execute_scenario_batch

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_builder_materializes_spot_rule() -> None:
    payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
    scenario = (
        ScenarioBuilder("spot-up", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z")
        .add_spot_shift("percentage", 0.10, underlying_name="UND_A HK")
        .build()
    )
    materialized = materialize_request(payload, scenario)
    underlying = materialized["UnwindMapRaw"]["underlyings"][0]
    original_spot = payload["UnwindMapRaw"]["underlyings"][0]["spot"]
    assert underlying["spot"] == original_spot * 1.10
    assert materialized["MarketDataSnapshot"]["spot_data"][0]["value"] == original_spot * 1.10


def test_current_report_is_virtual() -> None:
    scenario = ScenarioBuilder.current_report().build()
    assert scenario["scenario_kind"] == "virtual"
    assert scenario["market_data_source"] == "current"


def test_batch_persists_instance_and_parquet_partition(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
    scenario = ScenarioBuilder("unit-scenario", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z").build()
    store = RiskCubeStore(":memory:", tmp_path / "riskcube")
    summary = execute_scenario_batch([("case-1", payload)], scenario, store, batch_id="batch-1", version="v1")
    assert summary["instrument_count"] == 1
    assert summary["cell_count"] == 4
    assert len(summary["partitions"]) == 1
    assert Path(summary["partitions"][0]).exists()
    assert store.connection.execute("select count(*) from riskcube_instances where status = 'COMPLETED'").fetchone()[0] == 1
    assert store.connection.execute("select count(*) from riskcube_cells where scenario_id = ?", [summary["scenario_id"]]).fetchone()[0] == 4
    store.close()


def test_configurable_s3_endpoint(monkeypatch) -> None:
    import duckdb

    from riskcube_mcp.gcs import configure_duckdb_gcs, gcs_status

    monkeypatch.setenv("S3_API_KEY", "test-key")
    monkeypatch.setenv("S3_API_SECRET", "test-secret")
    monkeypatch.setenv("S3_BUCKET_NAME", "s3://test-bucket")
    monkeypatch.setenv("S3_ENDPOINT", "storage.example.test")
    connection = duckdb.connect(":memory:")
    configure_duckdb_gcs(connection)
    assert gcs_status()["endpoint"] == "storage.example.test"
    assert gcs_status()["credentials_present"] is True
    connection.close()
