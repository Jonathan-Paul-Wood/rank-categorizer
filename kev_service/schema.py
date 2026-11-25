"""Controller validation mirrors the pinned upstream API without importing its ML stack."""
import json
import math
import random
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JsonValue = None
    criteria: dict[str, JsonValue] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JsonValue = None
    criteria: dict[str, JsonValue] = Field(min_length=1, max_length=255)


class Score(BaseModel):
    type: Literal["score"]
    instructions: JsonValue = None
    criteria: list[JsonValue] = Field(min_length=1, max_length=255)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    state: JsonValue
    model: str = "kev-latest"
    questions: dict[str, Question] = Field(min_length=1)


class TrainOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=2, ge=1, le=20, strict=True)
    learning_rate: float = Field(default=2e-5, gt=0, le=0.001, allow_inf_nan=False)
    batch_size: int = Field(default=1, ge=1, le=8, strict=True)
    accumulation: int = Field(default=8, ge=1, le=128, strict=True)
    seed: int = Field(default=0, ge=0, le=2147483647, strict=True)


def strict_json(text):
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON number is out of range")
        return result
    return json.loads(text, parse_constant=invalid, parse_float=finite)


def validate_dataset(raw: bytes, max_records: int, max_questions: int) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = strict_json(line)
            if not isinstance(obj, dict) or set(obj) - {"state", "questions"}:
                raise ValueError("each row must contain only state and questions")
            req = DecisionRequest.model_validate(obj)
            if len(req.questions) > max_questions:
                raise ValueError("too many questions")
            cleaned = {"state": req.state, "questions": {}}
            for name, question in req.questions.items():
                supplied = obj["questions"][name]
                if set(supplied) - {"type", "instructions", "criteria", "label"}:
                    raise ValueError("unsupported question fields")
                if "label" not in supplied:
                    raise ValueError(f"question {name!r} needs a label")
                label = supplied["label"]
                if question.type == "noul" and type(label) is not bool:
                    raise ValueError("noul label must be true or false")
                if question.type == "choice" and (not isinstance(label, str) or label not in question.criteria):
                    raise ValueError("choice label must name a criterion")
                if question.type == "score" and (type(label) is not int or not 0 <= label < len(question.criteria)):
                    raise ValueError("score label must be an integer criterion index")
                cleaned["questions"][name] = {**question.model_dump(), "label": label}
            records.append(cleaned)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ValueError(f"invalid JSONL record at line {line_no}: {type(exc).__name__}") from exc
        if len(records) > max_records:
            raise ValueError(f"at most {max_records} records allowed")
    if len(records) < 5:
        raise ValueError("at least five labelled records required")
    return records


def split_dataset(records, seed):
    # Keep identical states together: repeated questions about one document cannot leak across partitions.
    groups = {}
    for record in records:
        key = json.dumps(record["state"], sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    if len(keys) < 5:
        raise ValueError("at least five distinct states required for a held-out split")
    random.Random(seed).shuffle(keys)
    evaluation = set(keys[:max(1, len(keys) // 5)])
    return ([r for key in keys if key not in evaluation for r in groups[key]],
            [r for key in keys if key in evaluation for r in groups[key]])

