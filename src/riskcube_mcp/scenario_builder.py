"""Scenario definitions and deterministic market-data materialization helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from typing import Any


class ScenarioBuilder:
    """Build a serializable scenario definition for batch pricing.

    A scenario may carry a materialized market-data snapshot or a rule set that
    is applied to a request template at execution time. The builder preserves
    the original attachment-compatible manipulation vocabulary.
    """

    def __init__(
        self,
        scenario_name: str,
        base_market_data_datetime: datetime | str,
        trade_repository_snapshot_datetime: datetime | str,
        *,
        scenario_id: str | None = None,
        scenario_version: str = "1",
        materialization_mode: str = "rules",
    ) -> None:
        self.scenario: dict[str, Any] = {
            "scenario_id": scenario_id,
            "scenario_version": scenario_version,
            "scenario_name": scenario_name,
            "scenario_kind": "rule",
            "materialization_mode": materialization_mode,
            "base_market_data_datetime": _iso(base_market_data_datetime),
            "trade_repository_snapshot_datetime": _iso(trade_repository_snapshot_datetime),
            "market_data_manipulations": [],
            "ir_curve_assignment": [],
            "metadata": {},
        }

    @classmethod
    def current_report(cls, scenario_name: str = "CURRENT_REPORT") -> ScenarioBuilder:
        now = datetime.now(UTC)
        builder = cls(
            scenario_name,
            now,
            now,
            scenario_id="VIRTUAL_CURRENT_REPORT",
            scenario_version="1",
            materialization_mode="current_market",
        )
        builder.scenario["scenario_kind"] = "virtual"
        builder.scenario["market_data_source"] = "current"
        builder.set_description("Virtual current-market report scenario for all instruments.")
        return builder

    def set_description(self, description: str) -> ScenarioBuilder:
        self.scenario["description"] = description
        return self

    def set_request_template(self, request_template: dict[str, Any]) -> ScenarioBuilder:
        self.scenario["request_template"] = copy.deepcopy(request_template)
        return self

    def set_market_data_snapshot(self, snapshot: dict[str, Any]) -> ScenarioBuilder:
        self.scenario["scenario_kind"] = "materialized"
        self.scenario["materialization_mode"] = "snapshot"
        self.scenario["market_data_snapshot"] = copy.deepcopy(snapshot)
        return self

    def add_manipulation(self, manipulation_type: str, **fields: Any) -> ScenarioBuilder:
        item = {"type": manipulation_type, **fields}
        self.scenario["market_data_manipulations"].append(item)
        return self

    def add_spot_shift(self, shift_type: str, value: float, underlying_name: str | None = None, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("spot_shift", shift_type=shift_type, value=value, underlying_name=underlying_name, filter_conditions=filter_conditions)

    def add_fx_manipulation(self, shift_type: str, value: float, currency_pair: str, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("fx_manipulation", shift_type=shift_type, value=value, currency_pair=currency_pair, filter_conditions=filter_conditions)

    def add_ir_manipulation(self, shift_type: str, value: float, curve_name: str, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("ir_manipulation", shift_type=shift_type, value=value, curve_name=curve_name, filter_conditions=filter_conditions)

    def add_dividend_manipulation(self, shift_type: str, value: float, underlying_name: str, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("dividend_manipulation", shift_type=shift_type, value=value, underlying_name=underlying_name, filter_conditions=filter_conditions)

    def add_vol_manipulation(self, shift_type: str, value: float, underlying_name: str, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("vol_manipulation", shift_type=shift_type, value=value, underlying_name=underlying_name, filter_conditions=filter_conditions)

    def add_fx_vol_manipulation(self, shift_type: str, value: float, currency_pair: str, filter_conditions: dict[str, Any] | None = None) -> ScenarioBuilder:
        return self.add_manipulation("fx_vol_manipulation", shift_type=shift_type, value=value, currency_pair=currency_pair, filter_conditions=filter_conditions)

    def add_ir_curve_assignment(self, curve_name: str, filter_conditions: dict[str, Any]) -> ScenarioBuilder:
        self.scenario["ir_curve_assignment"].append({"curve_name": curve_name, "filter_conditions": filter_conditions})
        return self

    def build(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.scenario)
        if not payload.get("scenario_id"):
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["scenario_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:24]
        return payload


def _iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def _shift(value: float, shift_type: str, amount: float) -> float:
    return value * (1.0 + amount) if shift_type in {"percentage", "relative"} else value + amount


def _matches_filter(request: dict[str, Any], conditions: dict[str, Any] | None) -> bool:
    if not conditions:
        return True
    instrument = request.get("InstrumentKey", {})
    for key, expected in conditions.items():
        if instrument.get(key) != expected and request.get(key) != expected:
            return False
    return True


def materialize_request(request: dict[str, Any], scenario: dict[str, Any], *, market_data_resolver: Any | None = None) -> dict[str, Any]:
    """Apply scenario rules to a request payload without mutating the input."""
    materialized = copy.deepcopy(request)
    if scenario.get("market_data_snapshot"):
        materialized["MarketDataSnapshot"] = copy.deepcopy(scenario["market_data_snapshot"])
    elif scenario.get("market_data_source") == "current" and market_data_resolver is not None:
        materialized["MarketDataSnapshot"] = copy.deepcopy(market_data_resolver(materialized))

    market_date = scenario.get("market_data_datetime") or scenario.get("base_market_data_datetime")
    if market_date:
        market_date_text = str(market_date).replace("Z", "")[:10]
        materialized.setdefault("parameters", {})["eval_datetime"] = market_date_text
        for rfk in materialized.get("RiskFactorKeys", []):
            if rfk.get("type") in {"Spot", "InterestRate", "FXSpot", "Dividend"} or rfk.get("temporal_role") == "ValuationDate":
                rfk["date"] = market_date_text

    underlyings = materialized.get("UnwindMapRaw", {}).get("underlyings", [])
    for manipulation in scenario.get("market_data_manipulations", []):
        if not _matches_filter(materialized, manipulation.get("filter_conditions")):
            continue
        kind = manipulation.get("type")
        shift_type = manipulation.get("shift_type", "absolute")
        amount = float(manipulation.get("value", 0.0))
        target_name = manipulation.get("underlying_name")
        if kind in {"spot_shift", "dividend_manipulation", "vol_manipulation"}:
            for underlying in underlyings:
                if target_name and underlying.get("name") != target_name:
                    continue
                if kind == "spot_shift":
                    underlying["spot"] = _shift(float(underlying["spot"]), shift_type, amount)
                    for point in materialized.get("MarketDataSnapshot", {}).get("spot_data", []):
                        if point.get("rfk", {}).get("underlying") == underlying.get("name"):
                            point["value"] = underlying["spot"]
                elif kind == "vol_manipulation":
                    for point in materialized.get("MarketDataSnapshot", {}).get("vol_data", []):
                        if point.get("rfk", {}).get("underlying") == underlying.get("name"):
                            point["value"] = _shift(float(point["value"]), shift_type, amount)
                else:
                    materialized.setdefault("ScenarioAdjustments", {}).setdefault("dividend", {})[underlying["name"]] = amount
        elif kind == "fx_manipulation":
            pair = manipulation.get("currency_pair")
            for point in materialized.get("MarketDataSnapshot", {}).get("fx_data", []):
                if point.get("rfk", {}).get("currency_pair") == pair:
                    point["value"] = _shift(float(point["value"]), shift_type, amount)
        elif kind == "ir_manipulation":
            curve = manipulation.get("curve_name")
            for point in materialized.get("MarketDataSnapshot", {}).get("ir_data", []):
                if not curve or point.get("rfk", {}).get("curve_name") == curve or point.get("rfk", {}).get("currency") == curve:
                    point["value"] = _shift(float(point["value"]), shift_type, amount)
    materialized.setdefault("ScenarioMetadata", {})["scenario_id"] = scenario.get("scenario_id")
    materialized["ScenarioMetadata"]["scenario_version"] = scenario.get("scenario_version", "1")
    return materialized
