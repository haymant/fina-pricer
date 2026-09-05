from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'data' / 'attachment_sample.json').read_text())
ENDPOINT = os.getenv('MCP_URL', 'https://fina-pricer.vercel.app/mcp')
OUTPUT = ROOT / 'data' / 'live_dated_scenarios_report.json'


def extract_json(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    content_type = response.headers.get('content-type', '')
    if 'application/json' in content_type and 'text/event-stream' not in content_type:
        return response.json()
    for line in response.text.splitlines():
        if line.startswith('data: '):
            return json.loads(line[6:])
    raise RuntimeError(f'No JSON-RPC payload in response: {response.status_code} {response.text[:1000]}')


class LiveMCP:
    def __init__(self, url: str) -> None:
        self.url = url
        self.session = requests.Session()
        self.headers = {
            'content-type': 'application/json',
            'accept': 'application/json, text/event-stream',
        }
        self.request_id = 0
        self.session_id: str | None = None

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.request_id += 1
        headers = dict(self.headers)
        if self.session_id:
            headers['mcp-session-id'] = self.session_id
        response = self.session.post(
            self.url,
            headers=headers,
            json={'jsonrpc': '2.0', 'id': self.request_id, 'method': method, 'params': params or {}},
            timeout=180,
        )
        self.session_id = response.headers.get('mcp-session-id', self.session_id)
        message = extract_json(response)
        if 'error' in message:
            raise RuntimeError(f'{method} failed: {message["error"]}')
        return message

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self.call('tools/call', {'name': name, 'arguments': arguments or {}})['result']
        text_items = [item.get('text') for item in result.get('content', []) if item.get('type') == 'text']
        if not text_items:
            return result
        text = text_items[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def scenario(key: str, date: str) -> dict[str, Any]:
    return {
        'scenario_id': key,
        'scenario_name': f'Live dynamic rules at {date}',
        'scenario_version': '1',
        'description': 'Live Vercel verification: +5% relative spot and +2 volatility points.',
        'scenario_kind': 'rule',
        'materialization_mode': 'rules',
        'base_market_data_datetime': f'{date}T00:00:00Z',
        'trade_repository_snapshot_datetime': f'{date}T00:00:00Z',
        'market_data_manipulations': [
            {'type': 'spot_shift', 'shift_type': 'percentage', 'value': 0.05, 'underlying_name': 'UND_A HK'},
            {'type': 'vol_manipulation', 'shift_type': 'absolute', 'value': 0.02, 'underlying_name': 'UND_A HK'},
        ],
    }


def main() -> None:
    client = LiveMCP(ENDPOINT)
    initialize = client.call('initialize', {
        'protocolVersion': '2025-03-26',
        'capabilities': {},
        'clientInfo': {'name': 'live-dated-scenario-verifier', 'version': '1.0'},
    })
    tools = client.call('tools/list')
    prompts = client.call('prompts/list')
    versions_before = client.tool('version_list')
    base = client.tool('scenario_create', {'scenario': scenario('LIVE_DYNAMIC_RULES_20270518', '2027-05-18')})
    later = client.tool('scenario_create', {'scenario': scenario('LIVE_DYNAMIC_RULES_20270818', '2027-08-18')})
    scenarios_after = client.tool('scenario_list')

    requests = [{'case_id': 'LIVE_FCN_SAMPLE_001', 'request': BASE}]
    runs = []
    for scenario_key, version_key in (
        ('LIVE_DYNAMIC_RULES_20270518', 'live-market-date-v1-20270518'),
        ('LIVE_DYNAMIC_RULES_20270818', 'live-market-date-v2-20270818'),
    ):
        runs.append(client.tool('scenario_trigger', {
            'scenario_id': scenario_key,
            'version': version_key,
            'batch_id': 'LIVE_DYNAMIC_RULES_PROMOTION_20260905',
            'requests': requests,
        }))

    versions_after = client.tool('version_list')
    instance_query = client.tool('olap_query', {
        'sql': '''
            SELECT version_id, version_key, scenario_id, scenario_key,
                   instance_id, status, instrument_count, cell_count
            FROM riskcube_instances
            WHERE batch_id = 'LIVE_DYNAMIC_RULES_PROMOTION_20260905'
            ORDER BY version_id, scenario_id
        '''
    })
    cell_query = client.tool('olap_query', {
        'sql': '''
            SELECT version_id, version_key, scenario_id, scenario_key,
                   instrument_id, MAX(pv_amount) AS pv_amount,
                   MAX(price_pct_of_notional) AS price_pct_of_notional,
                   MAX(CASE WHEN type = 'Spot' THEN CAST(sensitivities_json->>'delta' AS DOUBLE) END) AS delta,
                   MAX(CASE WHEN type = 'Volatility' THEN CAST(sensitivities_json->>'vega' AS DOUBLE) END) AS vega,
                   MAX(CASE WHEN type = 'InterestRate' THEN CAST(sensitivities_json->>'rho' AS DOUBLE) END) AS rho,
                   MAX(CASE WHEN type = 'Spot' THEN CAST(sensitivities_json->>'theta' AS DOUBLE) END) AS theta
            FROM riskcube_cells
            WHERE batch_id = 'LIVE_DYNAMIC_RULES_PROMOTION_20260905'
            GROUP BY version_id, version_key, scenario_id, scenario_key, instrument_id
            ORDER BY version_id, scenario_id, instrument_id
        '''
    })
    report = {
        'endpoint': ENDPOINT,
        'initialize': initialize,
        'tool_names': [item.get('name') for item in tools.get('result', {}).get('tools', [])],
        'prompt_names': [item.get('name') for item in prompts.get('result', {}).get('prompts', [])],
        'versions_before': versions_before,
        'created_scenarios': [base, later],
        'scenarios_after': scenarios_after,
        'runs': runs,
        'versions_after': versions_after,
        'instance_query': instance_query,
        'cell_query': cell_query,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
