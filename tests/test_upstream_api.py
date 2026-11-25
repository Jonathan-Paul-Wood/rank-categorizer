"""Executed in the Docker test target; local controller-only installs skip this module."""
import json
from pathlib import Path

import pytest

from kev_service.schema import DecisionRequest

api = pytest.importorskip("kev.api")


def test_controller_schema_preserves_upstream_record():
    request = json.loads((Path(__file__).resolve().parents[1] / "examples" / "request.json").read_text())
    controller = DecisionRequest.model_validate(request)
    original = api.SystemOneRequest.model_validate(request)
    forwarded = api.SystemOneRequest.model_validate(controller.model_dump())
    assert api.to_record(original) == api.to_record(forwarded)
