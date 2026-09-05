from __future__ import annotations

import json
from pathlib import Path

from riskcube_mcp.scenario_builder import ScenarioBuilder, materialize_request
from riskcube_mcp.storage import RiskCubeStore, execute_scenario_batch

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((ROOT / 'data' / 'attachment_sample.json').read_text())
OUTPUT = ROOT / 'data' / 'dynamic_dated_scenarios_report.json'
STORE_ROOT = ROOT / 'data' / 'dynamic_dated_riskcube'


def build_scenario(key: str, market_date: str) -> dict:
    return (
        ScenarioBuilder(
            f'Dynamic rules at {market_date}',
            f'{market_date}T00:00:00Z',
            f'{market_date}T00:00:00Z',
            scenario_id=key,
            scenario_version='1',
        )
        .set_description('Dynamic relative spot and volatility rules evaluated at the scenario market-data date.')
        .add_spot_shift('percentage', 0.05, underlying_name='UND_A HK')
        .add_vol_manipulation('absolute', 0.02, underlying_name='UND_A HK')
        .build()
    )


def summarize(store: RiskCubeStore, summary: dict) -> dict:
    row = store.connection.execute(
        '''
        SELECT i.version_id, i.version_key, i.scenario_id, i.scenario_key,
               i.instance_id, i.status, i.instrument_count, i.cell_count,
               MAX(c.pv_amount) AS pv_amount,
               MAX(c.price_pct_of_notional) AS price_pct_of_notional,
               MAX(CASE WHEN c.type = 'Spot' THEN CAST(c.sensitivities_json->>'delta' AS DOUBLE) END) AS delta,
               MAX(CASE WHEN c.type = 'Spot' THEN CAST(c.sensitivities_json->>'gamma' AS DOUBLE) END) AS gamma,
               MAX(CASE WHEN c.type = 'Volatility' THEN CAST(c.sensitivities_json->>'vega' AS DOUBLE) END) AS vega,
               MAX(CASE WHEN c.type = 'InterestRate' THEN CAST(c.sensitivities_json->>'rho' AS DOUBLE) END) AS rho,
               MAX(CASE WHEN c.type = 'Spot' THEN CAST(c.sensitivities_json->>'theta' AS DOUBLE) END) AS theta
        FROM riskcube_instances i
        JOIN riskcube_cells c ON c.instance_id = i.instance_id
        WHERE i.instance_id = ?
        GROUP BY ALL
        ''',
        [summary['instance_id']],
    ).fetchone()
    columns = [item[0] for item in store.connection.description]
    result = dict(zip(columns, row))
    result['partition'] = summary['partitions'][0]
    return result


def main() -> None:
    store = RiskCubeStore(':memory:', STORE_ROOT)
    scenarios = [
        build_scenario('DYNAMIC_RULES_20270518', '2027-05-18'),
        build_scenario('DYNAMIC_RULES_20270818', '2027-08-18'),
    ]
    requests = [('FCN_SAMPLE_001', PAYLOAD)]
    runs = []
    materialization_checks = []
    for index, scenario in enumerate(scenarios, start=1):
        materialized = materialize_request(PAYLOAD, scenario)
        materialization_checks.append({
            'scenario_key': scenario['scenario_id'],
            'market_data_date': scenario['base_market_data_datetime'],
            'materialized_eval_datetime': materialized['parameters']['eval_datetime'],
            'materialized_spot': materialized['UnwindMapRaw']['underlyings'][0]['spot'],
            'materialized_volatility': materialized['MarketDataSnapshot']['vol_data'][0]['value'],
        })
        runs.append(execute_scenario_batch(
            requests,
            scenario,
            store,
            batch_id='DYNAMIC_RULES_PROMOTION_20260905',
            version=f'market-date-v{index}-{scenario["base_market_data_datetime"][:10]}',
        ))

    comparisons = [summarize(store, run) for run in runs]
    if len(comparisons) == 2:
        comparisons[1]['pv_change_vs_first'] = comparisons[1]['pv_amount'] - comparisons[0]['pv_amount']
        comparisons[1]['price_pct_change_vs_first'] = comparisons[1]['price_pct_of_notional'] - comparisons[0]['price_pct_of_notional']
        comparisons[1]['delta_change_vs_first'] = comparisons[1]['delta'] - comparisons[0]['delta']
        comparisons[1]['gamma_change_vs_first'] = comparisons[1]['gamma'] - comparisons[0]['gamma']
        comparisons[1]['vega_change_vs_first'] = comparisons[1]['vega'] - comparisons[0]['vega']
        comparisons[1]['theta_change_vs_first'] = comparisons[1]['theta'] - comparisons[0]['theta']

    report = {
        'description': 'Two persisted dynamic-rule scenarios with identical shocks and different market-data dates.',
        'rule_family': {
            'spot_shift': '+5% relative spot on UND_A HK',
            'vol_manipulation': '+2 vol points on UND_A HK',
            'request_count': len(requests),
        },
        'materialization_checks': materialization_checks,
        'runs': runs,
        'comparisons': comparisons,
        'catalogs': {
            'versions': store.list_versions(),
            'scenarios': store.list_scenarios(),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))
    store.close()


if __name__ == '__main__':
    main()
