import json
from pathlib import Path

from riskcube_mcp import PricingRequest, sensitivity

root = Path(__file__).parents[1]
request = PricingRequest.model_validate(
    json.loads((root / "data" / "attachment_sample.json").read_text())
)
result = sensitivity(request)
(root / "data" / "attachment_sample_result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "PV": result["PV"],
            "PV_stderr": result["PV_stderr"],
            "cells": len(result["RiskCube"]["cells"]),
        },
        indent=2,
    )
)
