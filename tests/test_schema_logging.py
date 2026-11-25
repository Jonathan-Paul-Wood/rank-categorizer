import json

import pytest

from kev_service.config import Settings
from kev_service.debug_log import DebugLog
from kev_service.schema import split_dataset, validate_dataset


def test_debug_default_and_settings():
    assert not Settings().debug_mode
    assert Settings(debug_mode=" TRUE ").debug_mode
    with pytest.raises(ValueError):
        Settings(debug_mode="yes")
    with pytest.raises(ValueError):
        Settings(device="cpu", dtype="bf16")
    assert Settings(device="cuda").dtype == "auto"


def test_state_group_split(dataset):
    records = dataset + dataset
    train, evaluation = split_dataset(records, 0)
    assert len(train) == 16 and len(evaluation) == 4
    assert not {r["state"] for r in train} & {r["state"] for r in evaluation}
    assert (train, evaluation) == split_dataset(records, 0)


def test_validation(dataset):
    raw = "\n".join(json.dumps(r) for r in dataset).encode()
    assert len(validate_dataset(raw, 100, 64)) == 10
    with pytest.raises(ValueError):
        validate_dataset(raw, 5, 64)
    with pytest.raises(ValueError):
        validate_dataset(b'{}\n', 100, 64)
    with pytest.raises(ValueError):
        split_dataset([dataset[0]] * 10, 0)


def test_all_training_question_types(dataset):
    for record in dataset:
        record["questions"].update(
            team={"type": "choice", "criteria": {"a": None, "b": "other"}, "label": "a"},
            priority={"type": "score", "criteria": ["low", "high"], "label": 1})
    raw = "\n".join(json.dumps(r) for r in dataset).encode()
    assert len(validate_dataset(raw, 100, 64)) == 10


@pytest.mark.parametrize("label", [True, -1, 2, "0"])
def test_bad_score_label(label):
    record = {"state": "x", "questions": {"s": {"type": "score", "criteria": ["a", "b"], "label": label}}}
    with pytest.raises(ValueError):
        validate_dataset((json.dumps(record) + "\n").encode() * 5, 100, 64)


def test_log_reset_even_disabled(tmp_path):
    (tmp_path / "traffic.jsonl").write_text("stale secret")
    log = DebugLog(tmp_path)
    log.emit("ignored", secret="value")
    log.close()
    assert (tmp_path / "traffic.jsonl").read_bytes() == b""


def test_redaction_and_truncation(tmp_path):
    log = DebugLog(tmp_path, True, ("configured-secret",), max_event_bytes=400)
    log.emit("response", request_id="id", headers={"Authorization": "Bearer unknown", "Set-Cookie": "hidden"},
             body={"api-key": "hidden", "message": "configured-secret here"})
    log.emit("large", body="x" * 5000)
    log.close()
    text = log.path.read_text()
    assert "configured-secret" not in text and "hidden" not in text and "unknown" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0]["request_id"] == "id"
    assert rows[1]["truncated"]


def test_log_bounded_and_fail_open(tmp_path):
    log = DebugLog(tmp_path, True, max_events=2, max_bytes=1000)
    for i in range(4):
        log.emit("test", i=i)
        log.queue.join()
    log.close()
    assert len(log.path.read_text().splitlines()) == 2
    invalid = tmp_path / "file"
    invalid.write_text("x")
    broken = DebugLog(invalid, True)
    broken.emit("anything")
    broken.close()
    assert not broken.enabled
