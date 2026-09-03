import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from api.index import app
from riskcube_mcp import PricingRequest

print(type(app).__name__)
print(PricingRequest.__name__)
