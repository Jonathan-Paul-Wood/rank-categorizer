import asyncio
import json

import httpx
import pytest
import pytest_asyncio

from kev_service.app import create_app
from kev_service.catalog import MODELS
from kev_service.config import Settings
from kev_service.storage import write_json


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


class StubRunner:
    def __init__(self, settings, logger):
        self.settings = settings
        self.alive = False
        self.starts = 0
        self.fail_start = False
        self.fail_operation = None
        self.operation = None
        self.training_gate = asyncio.Event()
        self.training_gate.set()
        self.answer_gate = asyncio.Event()
        self.answer_gate.set()
        self.validating_gate = asyncio.Event()
        self.validating_gate.set()
        self.answer_started = asyncio.Event()
        self.validation_started = asyncio.Event()
        self.cancelled = False

    async def start(self, path):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("simulated reload failure")
        self.alive = True

    async def stop(self):
        self.alive = False

    async def request(self, body):
        self.answer_started.set()
        await self.answer_gate.wait()
        if self.fail_operation == "answer":
            raise TimeoutError("simulated timeout")
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.9}
            elif q["type"] == "choice":
                keys = list(q["criteria"])
                answers[key] = {"type": "choice", "choice": keys[0], "confidence": 0,
                                "probabilities": {key: 1 / len(keys) for key in keys}}
            else:
                answers[key] = {"type": "score", "score": 0.5, "confidence": 0,
                                "legend": {str(i): text for i, text in enumerate(q["criteria"])},
                                "probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}}
        return 200, {"model": body["model"], "answers": answers,
                     "usage": {"input_tokens": 5, "output_tokens": 20}, "latency_ms": 1}

    async def details(self):
        return {"name": "kev-latest", "description": "test model", "release_date": "2026-11-23"}

    async def run(self, operation, *args, **kwargs):
        self.operation = operation
        if self.fail_operation == operation:
            raise RuntimeError("simulated worker failure")
        if operation == "validate":
            self.validation_started.set()
            await self.validating_gate.wait()
        if operation == "train":
            try:
                await self.training_gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            directory = args[1]
            (directory / "staging" / "checkpoint").mkdir(parents=True)
            (directory / "staging" / "checkpoint" / "head.pt").write_bytes(b"test")
        if operation == "evaluate":
            directory = args[2]
            write_json(directory / "report.json", {"coverage": {"evaluated_records": 2, "rejected_records": 0}})

    async def close(self):
        await self.stop()


@pytest.fixture
def settings(tmp_path):
    bundle = tmp_path / "bundle"
    write_json(bundle / "bundle.json", {"model_id": "kev-0.8b", **MODELS["kev-0.8b"], "base_revision": "test"})
    (bundle / "checkpoint").mkdir()
    (bundle / "checkpoint" / "head.pt").write_bytes(b"test")
    return Settings(bundle_dir=bundle, data_dir=tmp_path / "data", debug_dir=tmp_path / "debug")


@pytest_asyncio.fixture
async def client(settings):
    app = create_app(settings, StubRunner)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await until(lambda: app.state.service.state == "ready")
            yield client, app.state.service


@pytest.fixture
def dataset():
    return [{"state": f"ticket {i}", "questions": {"urgent": {
        "type": "noul", "instructions": "Is it urgent?", "label": bool(i % 2)}}} for i in range(10)]


async def upload(client, dataset, options=None, **kwargs):
    return await client.post("/v1/fine-tunes", data={"name": "my-model", "options": json.dumps(options or {})},
                             files={"data": ("train.jsonl", "\n".join(json.dumps(r) for r in dataset),
                                             "application/x-ndjson")}, **kwargs)
