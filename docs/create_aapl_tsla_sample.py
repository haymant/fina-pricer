import json
from pathlib import Path

root = Path(__file__).parents[1]
base = json.loads((root / 'data' / 'attachment_sample.json').read_text())
base['InstrumentKey']['name'] = 'AAPL_TSLA_WORST_OF_FCN'
base['InstrumentKey']['isin'] = 'US_AAPL_TSLA_FCN'
base['UnwindMapRaw']['underlyings'] = [
    {**base['UnwindMapRaw']['underlyings'][0], 'name': 'AAPL US', 'spot': 200.0, 'strikePrice': 200.0, 'barrierPrice': 0.8, 'barriers': [{'direction': 'down', 'event': 'KI', 'level': 0.80, 'level_type': 'relative_initial', 'monitoring': 'global', 'observation_dates': [], 'rebate': 0.0}]},
    {**base['UnwindMapRaw']['underlyings'][0], 'name': 'TSLA US', 'spot': 250.0, 'strikePrice': 200.0, 'barrierPrice': 0.8, 'barriers': [{'direction': 'down', 'event': 'KI', 'level': 0.70, 'level_type': 'relative_initial', 'monitoring': 'global', 'observation_dates': [], 'rebate': 0.0}]},
]
base['RiskFactorKeys'] = [
    {'type': 'Spot', 'underlying': 'AAPL US'},
    {'type': 'Spot', 'underlying': 'TSLA US'},
    {'type': 'Volatility', 'underlying': 'AAPL US'},
    {'type': 'Volatility', 'underlying': 'TSLA US'},
    {'type': 'InterestRate', 'underlying': 'USD'},
]
base['MarketDataSnapshot']['spot_data'] = [
    {'rfk': {'underlying': 'AAPL US'}, 'value': 200.0},
    {'rfk': {'underlying': 'TSLA US'}, 'value': 250.0},
]
base['MarketDataSnapshot']['vol_data'] = [
    {'rfk': {'underlying': 'AAPL US'}, 'value': 0.30},
    {'rfk': {'underlying': 'TSLA US'}, 'value': 0.45},
]
base['UpdatedLifecycle']['adjusted_underlyings'] = [
    {'name': 'AAPL US', 'adjustment_factor': 1.0},
    {'name': 'TSLA US', 'adjustment_factor': 1.0},
]
base['parameters'].update({
    'payoff_type': 'fcn',
    'accrual': {'coupon_rate': 0.12, 'memory': True, 'observation_frequency': 'monthly', 'observations': 12, 'pay_if_ki': True},
    'volatility': 0.35,
    'correlation': [[1.0, 0.50], [0.50, 1.0]],
    'basket_method': 'worst_of',
    'bump_size': 0.01,
    'barriers': [],
})
(root / 'data' / 'basket_aapl_tsla.json').write_text(json.dumps(base, indent=2) + '\n')
print(root / 'data' / 'basket_aapl_tsla.json')
