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


def test_integer_catalog_ids_and_partition_layout(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
    store = RiskCubeStore(":memory:", tmp_path / "riskcube")
    scenario = ScenarioBuilder("catalog-scenario", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z", scenario_id="catalog-scenario").build()
    summary = execute_scenario_batch([("case-1", payload)], scenario, store, batch_id="promotion-1", version="release-2026-09")
    assert isinstance(summary["scenario_id"], int)
    assert isinstance(summary["version_id"], int)
    assert summary["scenario_key"] == "catalog-scenario"
    assert summary["version_key"] == "release-2026-09"
    assert store.connection.execute("SELECT count(*) FROM version_catalog").fetchone()[0] == 1
    assert store.connection.execute("SELECT count(*) FROM scenario_catalog").fetchone()[0] == 1
    assert "version_id=" in summary["partitions"][0]
    assert f"scenario_id={summary['scenario_id']}" in summary["partitions"][0]
    assert store.query_parquet(version=summary["version_id"], scenario_id=summary["scenario_id"])
    store.close()


def test_catalogs_reload_after_store_restart(tmp_path: Path) -> None:
    root = tmp_path / "riskcube"
    first = RiskCubeStore(":memory:", root)
    scenario = ScenarioBuilder("restart-scenario", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z", scenario_id="restart-scenario").build()
    version = first.register_version("release-restart")
    scenario_id = first.register_scenario(scenario)
    first.close()

    second = RiskCubeStore(":memory:", root)
    restored = second.get_scenario("restart-scenario")
    assert restored is not None
    assert restored["scenario_id"] == scenario_id
    assert second.list_versions()[0]["version_id"] == version["version_id"]
    next_scenario = ScenarioBuilder("next-scenario", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z", scenario_id="next-scenario").build()
    assert second.register_scenario(next_scenario) > scenario_id
    second.close()


def test_multi_scenario_promotion_reuses_version_id(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
    store = RiskCubeStore(":memory:", tmp_path / "riskcube")
    base = ScenarioBuilder("base", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z", scenario_id="base").build()
    shock = ScenarioBuilder("spot-up", "2027-05-18T00:00:00Z", "2027-05-18T00:00:00Z", scenario_id="spot-up").add_spot_shift("percentage", 0.10).build()
    first = execute_scenario_batch([("instrument-1", payload)], base, store, batch_id="promotion-1", version="release-multi")
    second = execute_scenario_batch([("instrument-1", payload)], shock, store, batch_id="promotion-1", version="release-multi")
    assert first["version_id"] == second["version_id"]
    assert first["scenario_id"] != second["scenario_id"]
    assert first["instance_id"] != second["instance_id"]
    assert store.connection.execute("SELECT count(*) FROM version_catalog WHERE version_key = 'release-multi'").fetchone()[0] == 1
    assert store.connection.execute("SELECT count(*) FROM riskcube_instances WHERE batch_id = 'promotion-1' AND status = 'COMPLETED'").fetchone()[0] == 2
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
