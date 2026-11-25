import asyncio

import httpx
import pytest

from conftest import StubRunner, until, upload
from kev_service.app import create_app
from kev_service.catalog import MODELS
from kev_service.storage import Registry, write_json

REQUEST = {"state": "charged twice", "questions": {
    "billing": {"type": "noul", "instructions": "About billing?"},
    "team": {"type": "choice", "criteria": {"billing": "payments", "other": None}},
    "priority": {"type": "score", "criteria": ["low", "high"]}}}


async def test_inference_compatibility(client):
    api, _ = client
    response = await api.post("/v1/systemone", json=REQUEST, headers={"x-typesafe-request-id": "caller-123"})
    assert response.status_code == 200
    assert response.headers["x-typesafe-request-id"] == "caller-123"
    assert response.json()["answers"]["billing"]["noul"] == 0.9
    assert set(response.json()) == {"model", "answers", "usage", "latency_ms"}
    models = (await api.get("/v1/models")).json()["models"]
    assert [m["name"] for m in models] == ["kev-0.8b", "kev-latest", "jev-latest"]
    assert (await api.get("/health/live")).status_code == 200


@pytest.mark.parametrize("body", [{}, {**REQUEST, "questions": {}},
                                 {**REQUEST, "questions": {"x": {"type": "choice", "criteria": {}}}}])
async def test_invalid_inference(client, body):
    assert (await client[0].post("/v1/systemone", json=body)).status_code == 422


async def test_model_selection(client):
    api, service = client
    assert (await api.post("/v1/systemone", json={**REQUEST, "model": "kev-4b"})).status_code == 404
    catalog = (await api.get("/v1/model-catalog")).json()["models"]
    assert len(catalog) == 4
    assert sum(m["selectable"] for m in catalog) == 1
    service.state = "loading"
    assert (await api.get("/health/ready")).status_code == 503
    assert (await api.post("/v1/systemone", json=REQUEST)).status_code == 503
    assert (await api.get("/v1/models")).status_code == 503


async def test_auth_and_limits(settings):
    settings.api_key = "secret"
    settings.max_request_bytes = 1024
    app = create_app(settings, StubRunner)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            for path in ("/v1/models", "/v1/model-catalog", "/v1/fine-tunes"):
                assert (await api.get(path)).status_code == 401
            assert (await api.get("/health/live")).status_code == 200
            headers = {"Authorization": "Bearer secret"}
            assert (await api.get("/v1/model-catalog", headers=headers)).status_code == 200
            assert (await api.post("/v1/systemone", content=b"x" * 1025, headers=headers)).status_code == 413
            async def chunks():
                yield b"x" * 800
                yield b"x" * 800
            assert (await api.post("/v1/systemone", content=chunks(), headers=headers)).status_code == 413


async def test_concurrency_limit(client):
    api, service = client
    service.settings.max_concurrency = 1
    service.runner.answer_gate.clear()
    first = asyncio.create_task(api.post("/v1/systemone", json=REQUEST))
    await service.runner.answer_started.wait()
    assert (await api.post("/v1/systemone", json=REQUEST)).status_code == 429
    service.runner.answer_gate.set()
    assert (await first).status_code == 200


async def test_training_success_and_custom_activation(client, dataset):
    api, service = client
    response = await upload(api, dataset)
    assert response.status_code == 202
    job = response.json()
    await until(lambda: service.active_job is None)
    status = (await api.get(job["status_url"])).json()
    assert status["status"] == "succeeded"
    assert status["evaluation"]["coverage"]["evaluated_records"] == 2
    assert service.runner.starts == 2
    assert service.state == "ready"
    catalog = (await api.get("/v1/model-catalog")).json()["models"]
    custom = catalog[-1]
    assert custom["selectable"] and not custom["active"] and custom["calibrated"] is False
    changed = service.settings.model_copy(update={"model_id": job["model_id"]})
    assert Registry(changed).active_path.name == "checkpoint"
    assert (await api.get("/v1/fine-tunes")).json()["jobs"][0]["id"] == job["id"]
    assert (await api.post(job["status_url"] + "/cancel")).json()["status"] == "succeeded"


async def test_drain_and_exclusivity(client, dataset):
    api, service = client
    service.runner.answer_gate.clear()
    request = asyncio.create_task(api.post("/v1/systemone", json=REQUEST))
    await service.runner.answer_started.wait()
    response = await upload(api, dataset)
    assert response.status_code == 202
    assert service.state == "draining" and service.runner.alive
    assert (await upload(api, dataset)).status_code == 409
    assert (await api.post("/v1/systemone", json=REQUEST)).status_code == 503
    service.runner.answer_gate.set()
    await request
    await until(lambda: service.active_job is None)
    assert service.state == "ready"


async def test_preflight_keeps_inference_online(client, dataset):
    api, service = client
    service.runner.validating_gate.clear()
    submission = asyncio.create_task(upload(api, dataset))
    await service.runner.validation_started.wait()
    assert (await api.post("/v1/systemone", json=REQUEST)).status_code == 200
    service.runner.validating_gate.set()
    assert (await submission).status_code == 202
    await until(lambda: service.active_job is None)


@pytest.mark.parametrize("operation", ["validate", "train", "evaluate"])
async def test_training_failures(client, dataset, operation):
    api, service = client
    service.runner.fail_operation = operation
    response = await upload(api, dataset)
    if operation == "validate":
        assert response.status_code == 422
        assert service.runner.starts == 1
    else:
        assert response.status_code == 202
        await until(lambda: service.active_job is None)
        assert service.job(response.json()["id"])["status"] == "failed"
        assert not service.registry.custom()
    assert service.state == "ready"


async def test_cancel_training(client, dataset):
    api, service = client
    service.runner.training_gate.clear()
    response = await upload(api, dataset)
    job = response.json()
    await until(lambda: service.state == "training")
    assert (await api.post(job["status_url"] + "/cancel")).status_code == 202
    await until(lambda: service.active_job is None)
    assert service.job(job["id"])["status"] == "cancelled"
    assert service.runner.cancelled and service.state == "ready"


async def test_cancel_while_draining(client, dataset):
    api, service = client
    service.runner.answer_gate.clear()
    request = asyncio.create_task(api.post("/v1/systemone", json=REQUEST))
    await service.runner.answer_started.wait()
    job = (await upload(api, dataset)).json()
    await api.post(job["status_url"] + "/cancel")
    await until(lambda: service.job(job["id"])["status"] == "cancelled")
    assert service.runner.starts == 1
    service.runner.answer_gate.set()
    await request
    await until(lambda: service.active_job is None)
    assert service.state == "ready"


async def test_reload_failure_keeps_management_online(client, dataset):
    api, service = client
    service.runner.fail_start = True
    job = (await upload(api, dataset)).json()
    await until(lambda: service.active_job is None)
    assert service.state == "failed"
    assert (await api.get("/health/ready")).status_code == 503
    assert (await api.get(job["status_url"])).json()["reload_error"]
    assert (await api.get("/v1/model-catalog")).status_code == 200


async def test_inference_failure_recovers(client):
    api, service = client
    service.runner.fail_operation = "answer"
    assert (await api.post("/v1/systemone", json=REQUEST)).status_code == 503
    await until(lambda: service.runner.starts == 2 and service.state == "ready")


async def test_restart_marks_interrupted(settings):
    job_id = "a" * 32
    write_json(settings.data_dir / "jobs" / job_id / "job.json", {
        "id": job_id, "model_id": f"ft-{job_id}", "created_at": "2026-01-01", "status": "training"})
    app = create_app(settings, StubRunner)
    async with app.router.lifespan_context(app):
        assert app.state.service.job(job_id)["status"] == "interrupted"


def test_incompatible_custom(settings):
    model_id = "ft-" + "a" * 32
    write_json(settings.data_dir / "models" / model_id / "model.json", {
        "id": model_id, "parent_model": "kev-4b", "parent_revision": MODELS["kev-4b"]["revision"],
        "base_revision": "test"})
    settings.model_id = model_id
    with pytest.raises(ValueError, match="incompatible"):
        Registry(settings)


async def test_bad_upload_options_and_labels(client, dataset):
    api, service = client
    assert (await upload(api, dataset, {"executable": "evil"})).status_code == 422
    dataset[0]["questions"]["urgent"]["label"] = 1
    assert (await upload(api, dataset)).status_code == 422
    assert service.runner.starts == 1
    assert (await api.get("/v1/fine-tunes/..%2F..%2Fsecret")).status_code == 404


async def test_custom_not_visible_until_evaluation(client, dataset):
    api, service = client
    service.runner.training_gate.clear()
    await upload(api, dataset)
    await until(lambda: service.state == "training")
    assert not service.registry.custom()
    service.runner.training_gate.set()
    await until(lambda: service.active_job is None)
    assert len(service.registry.custom()) == 1


async def test_27b_inference_only(client, dataset):
    api, service = client
    service.registry.bundled_id = "kev-27b"
    assert (await upload(api, dataset)).status_code == 422


async def test_upload_limit(client, dataset):
    api, service = client
    service.settings.max_upload_bytes = 1024
    response = await api.post("/v1/fine-tunes", data={"name": "test"},
                              files={"data": ("train.jsonl", b"x" * 1025)})
    assert response.status_code == 413


async def test_nonfinite_json_rejected(client):
    assert (await client[0].post("/v1/systemone", content=b'{"state":NaN,"questions":{}}')).status_code == 422


async def test_openapi_has_request_schema(client):
    schema = (await client[0].get("/openapi.json")).json()
    assert "requestBody" in schema["paths"]["/v1/systemone"]["post"]


async def test_shutdown_interrupts_active_training(client, dataset):
    api, service = client
    service.runner.training_gate.clear()
    job = (await upload(api, dataset)).json()
    await until(lambda: service.state == "training")
    await service.close()
    assert service.job(job["id"])["status"] == "interrupted"
    assert not service.runner.alive


async def test_traffic_logging_redacts_and_correlates(settings):
    import json
    settings.debug_mode = True
    settings.api_key = "configured-secret"
    app = create_app(settings, StubRunner)
    async with app.router.lifespan_context(app):
        await until(lambda: app.state.service.state == "ready")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            response = await api.post("/v1/systemone", json=REQUEST, headers={
                "Authorization": "Bearer configured-secret", "X-API-Key": "other-secret",
                "x-typesafe-request-id": "log-test"})
            assert response.status_code == 200
            await api.post("/v1/systemone", json={"state": "unauthorized body should be omitted"})
    text = (settings.debug_dir / "traffic.jsonl").read_text()
    assert "configured-secret" not in text and "other-secret" not in text
    assert "unauthorized body should be omitted" not in text
    records = [json.loads(line) for line in text.splitlines()]
    traffic = [r for r in records if r.get("request_id") == "log-test"]
    assert [r["event"] for r in traffic] == ["api.request", "api.input", "api.response"]
    assert traffic[-1]["body"]["answers"]["billing"]["noul"] == 0.9
