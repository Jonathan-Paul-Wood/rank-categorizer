"""Best-effort bounded scratch capture; only the controller owns the file."""
import json
import os
import queue
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

SENSITIVE = {"authorization", "proxy_authorization", "cookie", "set_cookie", "x_api_key", "api_key",
             "access_token", "refresh_token", "token", "password", "secret", "kev_api_key"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def redact(value, secrets=(), depth=0):
    if depth > 40:
        return "[depth limit]"
    if isinstance(value, dict):
        return {redact(str(k), secrets, depth + 1): "[REDACTED]"
                if str(k).lower().replace("-", "_") in SENSITIVE else redact(v, secrets, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets, depth + 1) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[omitted]"


class DebugLog:
    def __init__(self, directory, enabled=False, secrets=(), *, max_events=100,
                 max_event_bytes=1024 * 1024, max_bytes=30 * 1024 * 1024):
        self.path = Path(directory) / "traffic.jsonl"
        self.enabled, self.secrets = enabled, secrets
        self.max_events, self.max_event_bytes, self.max_bytes = max_events, max_event_bytes, max_bytes
        self.records = deque()
        self.queue = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.thread = None
        self.warned = False
        self.dropped = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.with_suffix(".tmp").unlink(missing_ok=True)
            self.path.write_bytes(b"")  # Also clear stale capture when disabled.
            self.path.chmod(0o600)
            if enabled:
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()
        except Exception:
            self._fail()

    def _fail(self):
        self.enabled = False
        if not self.warned:
            print("Debug capture unavailable; service continues.", file=sys.stderr)
            self.warned = True

    def emit(self, event, **fields):
        if not self.enabled or self.stop.is_set():
            return
        try:
            item = redact({"timestamp": now(), "event": event, **fields}, self.secrets)
            line = json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            if len(line) > self.max_event_bytes:
                item = {"timestamp": item["timestamp"], "event": event, "truncated": True,
                        "original_size_bytes": len(line), "request_id": item.get("request_id"),
                        "job_id": item.get("job_id")}
                line = json.dumps(item).encode() + b"\n"
            self.queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self._fail()

    def _persist(self, line):
        self.records.append(line)
        while len(self.records) > self.max_events or sum(map(len, self.records)) > self.max_bytes:
            self.records.popleft()
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.writelines(self.records)
        temporary.replace(self.path)

    def _loop(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                line = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._persist(line)
                if self.dropped:
                    dropped, self.dropped = self.dropped, 0
                    self._persist(json.dumps({"timestamp": now(), "event": "logging.dropped_events",
                                              "count": dropped}).encode() + b"\n")
            except Exception:
                self._fail()
            finally:
                self.queue.task_done()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
