import json
from collections import defaultdict
from pathlib import Path

report = json.loads(Path("data/custom_barrier_demo_report.json").read_text())
groups = defaultdict(list)
for row in report["results"]:
    parts = row["case_id"].split("_")
    moneyness, tenor, memory = parts[0], parts[1], parts[-1]
    barrier = "_".join(parts[2:-1])
    groups[(barrier, memory)].append(row)

print("condition,pv_mean,pv_min,pv_max,spot_delta_mean,spot_gamma_mean,methods")
for key, rows in sorted(groups.items()):
    cells = [
        next(
            cell for cell in row["risk_cube"]["cells"] if cell["rfk"]["type"] == "Spot"
        )
        for row in rows
    ]
    values = [row["pv"] for row in rows]
    deltas = [cell["sensitivities"]["delta"] for cell in cells]
    gammas = [cell["sensitivities"]["gamma"] for cell in cells]
    methods = sorted({method for row in rows for method in row["methods"]})
    print(
        f"{key[0]}|memory={key[1]},{sum(values) / len(values):.8f},{min(values):.8f},{max(values):.8f},{sum(deltas) / len(deltas):.8f},{sum(gammas) / len(gammas):.8f},{';'.join(methods)}"
    )

print("\ncase_id,pv,spot_delta,spot_gamma,theta,barrier_events,coupon_memory_carry")
for row in report["results"]:
    spot = next(
        cell for cell in row["risk_cube"]["cells"] if cell["rfk"]["type"] == "Spot"
    )["sensitivities"]
    print(
        f"{row['case_id']},{row['pv']:.8f},{spot['delta']:.8f},{spot['gamma']:.8f},{spot['theta']:.8f},{len(row['explainability']['barrier_events'])},{row['explainability']['coupon_state']['memory_carry']:.8f}"
    )
