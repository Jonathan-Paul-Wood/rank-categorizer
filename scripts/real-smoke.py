"""Real offline inference -> training -> persistent custom activation; uses an isolated Docker volume."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--image", default="kev-ranker:1.0-cpu")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--timeout", type=int, default=3600)
args = parser.parse_args()
name = "kev-check-" + uuid.uuid4().hex[:12]
volume = name + "-data"


def docker(*parts, **kwargs):
    return subprocess.run(["docker", *parts], check=True, text=True, encoding="utf-8", errors="replace",
                          capture_output=True, **kwargs).stdout


def request(path, body=None, content_type="application/json"):
    encoded = base64.b64encode(body or b"").decode()
    code = ("import base64,json; from urllib.request import Request,urlopen; "
            f"r=Request('http://127.0.0.1:8009{path}',"
            f"base64.b64decode('{encoded}') if {body is not None!r} else None,"
            f"{{'Content-Type':{content_type!r}}}); "
            "print(urlopen(r,timeout=600).read().decode())")
    return json.loads(docker("exec", name, "python", "-c", code))


def wait_ready():
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if request("/health/ready")["status"] == "ready":
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(2)
    raise RuntimeError("container did not become ready")


def start(model):
    command = ["run", "-d", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:size=1g,mode=1777", "--tmpfs", "/debug:size=64m,uid=10001,gid=10001,mode=0700",
               "--mount", f"type=volume,source={volume},target=/data", "-e", f"KEV_MODEL={model}",
               "-e", "KEV_CPU_THREADS=2", "-e", "DEBUG_MODE=true", "-e", "KEV_INFERENCE_TIMEOUT=600"]
    if args.cuda:
        command += ["--gpus", "all", "-e", "KEV_DEVICE=cuda"]
    docker(*command, args.image)
    wait_ready()


try:
    docker("volume", "create", volume)
    start("kev-0.8b")
    body = (ROOT / "examples" / "request.json").read_bytes()
    assert "answers" in request("/v1/systemone", body)
    print("Offline standard inference passed", flush=True)
    boundary = "kev-test-boundary"
    dataset = (ROOT / "examples" / "train.jsonl").read_bytes()
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\nsmoke\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="options"\r\n\r\n'
               '{"epochs":1,"accumulation":1}\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="data"; filename="train.jsonl"\r\n'
               'Content-Type: application/x-ndjson\r\n\r\n').encode() + dataset + f"\r\n--{boundary}--\r\n".encode()
    job = request("/v1/fine-tunes", payload, f"multipart/form-data; boundary={boundary}")
    deadline = time.monotonic() + args.timeout
    last_status = None
    while time.monotonic() < deadline:
        status = request(job["status_url"])
        if status["status"] != last_status:
            last_status = status["status"]
            print(f"Job: {last_status}", flush=True)
        if last_status in ("failed", "cancelled", "interrupted"):
            raise RuntimeError(json.dumps(status))
        if last_status == "succeeded":
            break
        time.sleep(2)
    else:
        raise TimeoutError("fine-tune timed out")
    assert status["evaluation"]["coverage"]["rejected_records"] == 0
    wait_ready()
    assert "answers" in request("/v1/systemone", body)
    print("Configured inference model restored", flush=True)
    docker("rm", "-f", name)
    start(job["model_id"])
    custom_request = json.loads(body)
    custom_request["model"] = job["model_id"]
    assert "answers" in request("/v1/systemone", json.dumps(custom_request).encode())
    catalog = request("/v1/model-catalog")["models"]
    assert any(m["id"] == job["model_id"] and m["active"] and not m["calibrated"] for m in catalog)
    print("Offline fine-tune, evaluation, persistence, and custom inference passed", flush=True)
except BaseException as error:
    if isinstance(error, subprocess.CalledProcessError):
        print(error.stderr or error.stdout, file=sys.stderr)
    subprocess.run(["docker", "logs", "--tail", "120", name], check=False)
    raise
finally:
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(["docker", "volume", "rm", volume], check=False, capture_output=True)
