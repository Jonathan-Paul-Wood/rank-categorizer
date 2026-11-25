"""Standard-library client: python scripts/smoke-test.py [--base-url URL] [--api-key KEY]."""
import argparse
import json
import os
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default=os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009"))
parser.add_argument("--api-key", default=os.getenv("KEV_API_KEY", ""))
args = parser.parse_args()
headers = {"Content-Type": "application/json"}
if args.api_key:
    headers["Authorization"] = f"Bearer {args.api_key}"
with urlopen(args.base_url + "/health/ready", timeout=10) as response:
    assert json.load(response)["status"] == "ready"
payload = {"state": "I was charged twice. Please refund the extra charge.", "questions": {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments", "shipping": "Delivery"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "high"]}}}
request = Request(args.base_url + "/v1/systemone", json.dumps(payload).encode(), headers)
with urlopen(request, timeout=600) as response:
    output = json.load(response)
    assert response.headers["x-typesafe-request-id"]
    assert set(output["answers"]) == {"billing", "team", "urgency"}
    assert 0 <= output["answers"]["billing"]["noul"] <= 1
    assert abs(sum(output["answers"]["team"]["probabilities"].values()) - 1) < 0.02
    print(json.dumps(output, indent=2))

