import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = "kev-0.8b"
    bundle_dir: Path = Path("/opt/models")
    data_dir: Path = Path("/data")
    debug_dir: Path = Path("/debug")
    debug_mode: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["auto", "fp32", "bf16"] = "auto"
    cpu_threads: int = Field(default=6, ge=1, le=256)
    host: str = "0.0.0.0"
    port: int = Field(default=8009, ge=1, le=65535)
    api_key: str = ""
    max_request_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    max_upload_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    max_records: int = Field(default=10000, ge=5)
    max_questions: int = Field(default=64, ge=1, le=1024)
    max_concurrency: int = Field(default=8, ge=1, le=256)
    inference_timeout: float = Field(default=300, gt=0)
    startup_timeout: float = Field(default=600, gt=0)
    training_timeout: float = Field(default=86400, gt=0)
    validation_timeout: float = Field(default=300, gt=0)

    @field_validator("debug_mode", mode="before")
    @classmethod
    def debug_bool(cls, value):
        if isinstance(value, str):
            if value.lower().strip() not in ("true", "false"):
                raise ValueError("DEBUG_MODE must be true or false")
            return value.lower().strip() == "true"
        return value

    @field_validator("api_key")
    @classmethod
    def printable_key(cls, value):
        if value and (len(value) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in value)):
            raise ValueError("KEV_API_KEY must be 1-512 printable ASCII characters without spaces")
        return value

    @model_validator(mode="after")
    def check_device(self):
        if self.device == "cpu" and self.dtype == "bf16":
            raise ValueError("CPU uses fp32; bf16 requires CUDA")
        if not self.model_id:
            raise ValueError("KEV_MODEL cannot be empty")
        return self

    @classmethod
    def from_env(cls):
        mapping = {
            "model_id": "KEV_MODEL", "bundle_dir": "KEV_BUNDLE_DIR", "data_dir": "KEV_DATA_DIR",
            "debug_dir": "KEV_DEBUG_DIR", "debug_mode": "DEBUG_MODE", "device": "KEV_DEVICE",
            "dtype": "KEV_DTYPE", "cpu_threads": "KEV_CPU_THREADS", "host": "KEV_HOST", "port": "KEV_PORT",
            "api_key": "KEV_API_KEY", "max_request_bytes": "KEV_MAX_REQUEST_BYTES",
            "max_upload_bytes": "KEV_MAX_UPLOAD_BYTES", "max_records": "KEV_MAX_RECORDS",
            "max_questions": "KEV_MAX_QUESTIONS", "max_concurrency": "KEV_MAX_CONCURRENCY",
            "inference_timeout": "KEV_INFERENCE_TIMEOUT", "startup_timeout": "KEV_STARTUP_TIMEOUT",
            "training_timeout": "KEV_TRAINING_TIMEOUT", "validation_timeout": "KEV_VALIDATION_TIMEOUT",
        }
        return cls(**{key: os.environ[name] for key, name in mapping.items() if name in os.environ})

