# Kev Ranker

Run one [Kev decision model](https://github.com/jaredpalmer/kev) behind a LAN API, and fine-tune it on your own labelled examples. CPU and NVIDIA CUDA images use the same API. Inference returns yes/no probabilities, choices, and scores; Kev does not generate chat completions.

Only one model resides in memory at a time. The image contains one standard checkpoint plus its required base weights and tokenizer. Custom models live in a persistent Docker volume and can be selected through `.env`. Startup, inference, and fine-tuning work offline after building the image.

## Quick start

Requirements: Linux AMD64 server, Docker Engine with Compose v2, and Python 3.12+ on the machine running the helper scripts. Windows builds require Docker Desktop in Linux container mode. The server only needs Python 3 for the deployment helper; model dependencies run inside Docker.

Start with 0.8B on CPU. Allow several GB for its weights and additional space for Docker layers, download caches, and checkpoints. CPU serving uses FP32 weights. Larger models require substantially more RAM; 27B contains about 51 GB of BF16 checkpoint weights and needs additional runtime memory. Fine-tuning needs more memory than inference. Hardware and document length determine practical speed.

From PowerShell:

```powershell
Copy-Item .env.example .env
# Edit .env for your server address and optional KEV_API_KEY.
python scripts/manage.py download
.\scripts\build-image.ps1 -OutputArchive kev-ranker-image.tar
```

From Linux:

```bash
cp .env.example .env
chmod 600 .env
python3 scripts/manage.py download
bash scripts/build-image.sh
bash scripts/deploy.sh
python3 scripts/smoke-test.py
```

The download command defaults to `KEV_BUNDLE_MODEL` from `.env`, or `kev-0.8b`. It first builds the CPU `model-tools` image, then downloads into `model-cache/<model-id>` and exports `model-cache/<model-id>.tar`. Interrupted Hugging Face downloads resume when repeated. The tar preserves Linux cache links when building from Windows. The build extracts only the chosen archive, verifies its SHA-256 file manifest, and includes that model in the image. Other downloaded models remain on the build host. Allow space for both the download cache and its uncompressed tar.

To build and transfer from Windows:

```powershell
ssh user@server 'mkdir -p ~/kev-ranker/scripts'
scp compose.yaml compose.cuda.yaml .env.example user@server:~/kev-ranker/
scp scripts/manage.py scripts/deploy.sh scripts/smoke-test.py user@server:~/kev-ranker/scripts/
scp -r kev_service user@server:~/kev-ranker/
scp kev-ranker-image.tar user@server:~/kev-ranker/
```

On the server, copy `.env.example` to `.env`, edit it, then run:

```bash
cd ~/kev-ranker
chmod 600 .env
bash scripts/deploy.sh --load --archive kev-ranker-image.tar
KEV_API_KEY='your-key' python3 scripts/smoke-test.py
```

Loading/deploying an existing image does not build, pull, or download anything unless `--pull` is explicitly supplied. Docker and Python must already be installed. Enable Docker at host boot; the container uses `restart: unless-stopped`. An unhealthy Docker health status does not itself restart a container.

## Models and CUDA

| Model ID | Download/build option | Fine-tuning |
| --- | --- | --- |
| `kev-0.8b` | Default | Supported |
| `kev-4b` | `--model kev-4b` | Supported |
| `kev-9b` | `--model kev-9b` | Supported |
| `kev-27b` | `--model kev-27b` | Inference only |

To switch standard models, download the new model, set both `KEV_BUNDLE_MODEL` and `KEV_MODEL` in `.env`, give `KEV_IMAGE` a new tag, rebuild, then deploy:

```bash
python3 scripts/manage.py download --model kev-4b
# Edit .env: KEV_BUNDLE_MODEL=kev-4b, KEV_MODEL=kev-4b, KEV_IMAGE=kev-ranker:4b-cpu
python3 scripts/manage.py build
python3 scripts/manage.py deploy
```

For NVIDIA, install a driver compatible with PyTorch's CUDA 12.8 wheels and NVIDIA Container Toolkit on the server. Set:

```dotenv
KEV_IMAGE=kev-ranker:1.0-cuda
KEV_TORCH_INDEX=https://download.pytorch.org/whl/cu128
KEV_DEVICE=cuda
KEV_DTYPE=auto
```

Then build and deploy using the same helper commands. They select `compose.cuda.yaml` when `KEV_DEVICE=cuda`. The `--cuda` build option is also available, but deployment settings in `.env` must still match. Automatic precision uses BF16 on supported CUDA hardware and FP32 otherwise; explicit `KEV_DTYPE=fp32` or `bf16` overrides it. Training uses BF16 autocast when the GPU supports it, otherwise FP32. An explicit CUDA device failure never falls back silently to CPU. CUDA fused kernels and CUDA graphs are disabled in this initial service.

If invoking Compose directly for CUDA, always include both files:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml up -d --no-build --pull never --wait --wait-timeout 600
```

## Inference API

Default URL: `http://SERVER_PRIVATE_IP:8009`. The container listens on `0.0.0.0:8009`; set `KEV_BIND_IP` to the server's private IP to limit the host interface. By default it publishes on all host interfaces. Use a firewall to restrict access to the intended LAN. Set `KEV_API_KEY` to require `Authorization: Bearer YOUR_KEY` on every `/v1` endpoint. Health endpoints remain unauthenticated. API traffic uses HTTP; use your own reverse proxy for TLS when needed.

```bash
curl http://SERVER_PRIVATE_IP:8009/v1/systemone \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_KEY' \
  --data-binary @examples/request.json
```

Omit the authorization header if `KEV_API_KEY` is empty. On Windows use `curl.exe`.

Request:

```json
{
  "model": "kev-latest",
  "state": "I was charged twice. Please refund the extra payment.",
  "questions": {
    "team": {"type": "choice", "instructions": "Who should handle this?", "criteria": {"billing": "Payments and refunds", "shipping": "Delivery"}},
    "urgent": {"type": "noul", "instructions": "Does this need immediate attention?"},
    "priority": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "normal", "high"]}
  }
}
```

- `state` is required and accepts any JSON value; objects and arrays are flattened by Kev.
- `questions` is a nonempty object mapping caller-chosen IDs to questions. Each question sees the shared state independently.
- `instructions` accepts JSON and describes the decision.
- `choice.criteria` maps option names to descriptions; descriptions can be null. `score.criteria` is an ordered array. Both allow 1–255 options.
- `noul` returns the probability of true. It optionally accepts `criteria` descriptions keyed by `false` and `true`.
- `model` defaults to `kev-latest`. The active ID, `kev-latest`, and `jev-latest` all select the same loaded checkpoint. Any other ID returns 404.

Illustrative response (values depend on the model):

```json
{
  "model": "kev-latest",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": {"billing": 0.9, "shipping": 0.1}},
    "urgent": {"type": "noul", "noul": 0.7},
    "priority": {"type": "score", "score": 1.4, "confidence": 0.2, "legend": {"0": "low", "1": "normal", "2": "high"}, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}
  },
  "usage": {"input_tokens": 100, "output_tokens": 160},
  "latency_ms": 125.5
}
```

Scores are the expected zero-based level, not necessarily an integer. Confidence uses upstream Kev's formulas. Usage counts serialized output tokens; the model performs decision inference rather than text generation. `latency_ms` is Kev's model timing, not total HTTP elapsed time.

Responses contain `x-typesafe-request-id` and `x-request-id`. A printable caller `x-typesafe-request-id` of at most 128 characters is retained; otherwise the service creates one.

| Endpoint | Response |
| --- | --- |
| `POST /v1/systemone` | Typed answers shown above |
| `GET /v1/models` | `{"models":[...]}` for the active ID and two compatibility aliases, with runtime details |
| `GET /v1/model-catalog` | Four standard entries plus completed custom entries; each has `id`, `selectable`, `bundled`, and `active` |
| `GET /health/live` | 200 while the controller is running |
| `GET /health/ready` | 200 when ready; otherwise 503 with state, selected model, and load error |
| `GET /docs` | Interactive OpenAPI documentation |

Errors use `{"detail":...}`. Common statuses: 401 invalid credentials, 404 inactive/unknown model or job, 409 overlapping training, 413 size limit, 422 invalid input or training preflight failure, 429 inference concurrency limit, and 503 loading/training/recovery/unavailable worker. Retry 429/503 with backoff. A failed inference worker is restarted after admitted requests drain. A reload failure leaves management APIs live; inspect logs and restart after correcting the problem.

Kev's token/context limits remain enforced. Overlong inference returns 422; this service does not silently truncate state. Leave generous CPU timeouts in clients.

## Fine-tuning

Fine-tuning genuinely updates adapter/head weights using supplied labels. There is no external LLM or synthetic label generation. It always starts from the bundled standard checkpoint, including when a custom model is active. Training data and checkpoints stay on the server.

1. Prepare a UTF-8 JSONL file. Each line contains `state` and `questions`, shaped like an inference request with a `label` on every question. Instructions and criteria belong in each question; there is no separate global instruction field. Reuse your task's instructions in future inference requests.
2. Submit the file and allowed training options.
3. Poll the returned status URL until completion. Inference is temporarily unavailable during draining, training, evaluation, and reload.
4. Inspect the held-out evaluation report, then explicitly select the resulting model ID if it suits your task.

Label examples:

```json
{"state":"Charged twice","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","shipping":"Delivery"},"label":"billing"},"angry":{"type":"noul","instructions":"Is the customer angry?","label":false},"priority":{"type":"score","instructions":"How urgent?","criteria":["low","normal","high"],"label":1}}}
```

`choice` labels must name an option; `noul` labels must be JSON booleans; `score` labels must be zero-based integers. Only `state`/`questions` and the documented question fields are accepted. Use at least five records with five distinct states. The included ten-row `examples/train.jsonl` is for functional smoke testing, not measuring useful accuracy improvements.

```bash
curl http://SERVER_PRIVATE_IP:8009/v1/fine-tunes \
  -H 'Authorization: Bearer YOUR_KEY' \
  -F 'name=support-routing' \
  -F 'options={"epochs":2,"learning_rate":0.00002,"batch_size":1,"accumulation":8,"seed":0}' \
  -F 'data=@examples/train.jsonl;type=application/x-ndjson'
```

`name` is a display name, 1–100 characters. `options` is a JSON object; omit it for defaults. The service generates IDs, and accepts no paths, executable names, or additional trainer arguments.

| Option | Default | Accepted range |
| --- | --- | --- |
| `epochs` | 2 | Integer 1–20 |
| `learning_rate` | 0.00002 | Finite number greater than 0, at most 0.001 |
| `batch_size` | 1 | Integer 1–8 |
| `accumulation` | 8 | Integer 1–128 |
| `seed` | 0 | Integer 0–2147483647 |

The JSONL upload, labels, and tokenizer/context fit are validated before pausing inference. Validation can take time; allow at least `KEV_VALIDATION_TIMEOUT` in the submitting client. No second GPU model is loaded for preflight. Identical states stay in one partition. A reproducible shuffle assigns approximately 80% of distinct states to training and 20% to evaluation. Repeated states can change the record-count ratio. Gradient checkpointing is enabled, and adapter architecture settings come from the standard checkpoint.

Successful submission returns HTTP 202:

```json
{
  "id": "0123456789abcdef0123456789abcdef",
  "name": "support-routing",
  "model_id": "ft-0123456789abcdef0123456789abcdef",
  "status": "draining",
  "parent_model": "kev-0.8b",
  "options": {"epochs":2,"learning_rate":0.00002,"batch_size":1,"accumulation":8,"seed":0},
  "train_records": 8,
  "evaluation_records": 2,
  "created_at": "2026-11-23T12:00:00.000Z",
  "updated_at": "2026-11-23T12:00:00.000Z",
  "status_url": "/v1/fine-tunes/0123456789abcdef0123456789abcdef",
  "error": null
}
```

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/fine-tunes` | `{"jobs":[...]}` ordered newest first |
| `GET /v1/fine-tunes/{id}` | Job fields; completed jobs also include `evaluation` from the upstream report |
| `POST /v1/fine-tunes/{id}/cancel` | 202 with job status; repeated calls for a terminal job return that status |

States progress through `draining`, `training`, `evaluating`, and `succeeded`. Cancellation can briefly show `cancelling` before `cancelled`. Failures become `failed`; controller restart marks unfinished jobs `interrupted`. Jobs do not automatically resume. Only one job is admitted at a time, including model restoration. Existing inference calls finish before the worker unloads. A cancellation during draining still waits for those requests before reloading.

Checkpoint and evaluation outputs are published together by an atomic directory rename. Failed or cancelled outputs are not selectable. Every held-out record must be evaluated before publication. Custom checkpoints use temperature 1.0 and are marked `calibrated:false`; fine-tuning invalidates the standard model's fitted calibration. Review accuracy, Brier/calibration metrics, and workload-specific behavior before setting automation thresholds. The evaluation set is not used for training or temperature fitting.

After completion, edit only `KEV_MODEL` to the returned `ft-...` ID, then recreate the container:

```bash
python3 scripts/manage.py deploy
```

The image's bundled standard model and revision must match the custom checkpoint's parent. A custom model trained from 4B requires the matching 4B image. Startup rejects missing/incompatible selections. Switching back to the bundled standard ID requires recreation, not rebuilding. Training never changes `.env` or automatically activates its output.

## Configuration and storage

Compose reads `.env`; helper scripts read it without executing shell code. Use simple `KEY=value` entries without variable interpolation. Process environment overrides packaging settings. Recreate the container after changing runtime settings. `KEV_BUNDLE_MODEL` is a build setting; `KEV_MODEL` selects the runtime standard/custom checkpoint.

| Variable | Default | Purpose |
| --- | --- | --- |
| `KEV_IMAGE` | `kev-ranker:1.0-cpu` | Image tag built, exported, or deployed |
| `KEV_BUNDLE_MODEL` | `kev-0.8b` | Standard model included during build |
| `KEV_MODEL` | `kev-0.8b` | Active standard/custom model ID |
| `KEV_TORCH_INDEX` | PyTorch CPU wheel index | Use `https://download.pytorch.org/whl/cu128` for CUDA |
| `KEV_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KEV_DTYPE` | `auto` | Inference: `auto`, `fp32`, or `bf16` (CUDA only) |
| `KEV_CPU_THREADS` | 6 | OpenMP/MKL thread count, 1–256 |
| `KEV_MEMORY_LIMIT` | `21g` | Docker container RAM cap for CPU and CUDA deployments; increase for larger models or fine-tuning |
| `KEV_BIND_IP` | `0.0.0.0` | Host interface for port publication |
| `KEV_HOST_PORT` | 8009 | Published host port |
| `KEV_API_KEY` | Empty | Optional bearer authentication |
| `DEBUG_MODE` | `false` | `true` enables bounded body/event capture |
| `KEV_MAX_REQUEST_BYTES` | 2097152 | Inference/other non-upload request limit |
| `KEV_MAX_UPLOAD_BYTES` | 33554432 | Dataset limit; multipart envelope has 64 KiB extra allowance |
| `KEV_MAX_RECORDS` | 10000 | Maximum labelled records per upload |
| `KEV_MAX_QUESTIONS` | 64 | Questions per record/request |
| `KEV_MAX_CONCURRENCY` | 8 | Admitted inference requests; excess gets 429 |
| `KEV_INFERENCE_TIMEOUT` | 300 | Internal inference HTTP timeout in seconds |
| `KEV_STARTUP_TIMEOUT` | 600 | Model load/readiness deadline; deploy wait deadline |
| `KEV_TRAINING_TIMEOUT` | 86400 | Deadline per training/evaluation subprocess, seconds |
| `KEV_VALIDATION_TIMEOUT` | 300 | Upload tokenizer preflight deadline, seconds |

For direct Python use, `KEV_HOST`/`KEV_PORT` default to `0.0.0.0`/8009, and `KEV_BUNDLE_DIR`, `KEV_DATA_DIR`, `KEV_DEBUG_DIR` default to `/opt/models`, `/data`, `/debug`. Compose fixes these container values. The controller runs exactly one process; do not use multiple Uvicorn workers against one data directory.

The `kev-ranker_kev-data` named volume contains `models/<model-id>/{model.json,checkpoint/,evaluation/}` and `jobs/<job-id>/{job.json,input.jsonl,train.jsonl,evaluation.jsonl,...}`. Inputs, reports, and incomplete staging outputs persist for diagnosis. They can contain your training data. Back up the volume; do not use `docker compose down -v` unless you intend to delete all custom models and jobs. No automatic data-retention cleanup is performed.

## Debugging and operations

`DEBUG_MODE=false` disables traffic capture. On every controller startup, `/debug/traffic.jsonl` is cleared even when capture is disabled. The default Compose debug directory is a 64 MiB tmpfs. Model reloads and training do not clear it.

When enabled, records contain UTC timestamps, event names, request/job IDs, input/output bodies, worker progress, errors, and timings. Multipart file content is omitted; upload filename, byte count, record count, and options are logged. Authorization/cookie/API-key fields and configured secret values are redacted. Arbitrary secrets inside application text cannot all be recognized; debug bodies may contain private business data.

Capture retains at most 100 events and 30 MiB, caps each event at 1 MiB, and replaces oversized content with a truncation marker. A bounded writer queue drops excess events and records a drop count when possible. The logger atomically replaces the JSONL file and fails open if storage is unavailable. Worker progress also appears in bounded Docker logs with configured-key redaction.

```bash
docker compose ps
docker compose logs -f --tail 100 kev
docker compose exec kev cat /debug/traffic.jsonl
docker compose cp kev:/debug/traffic.jsonl ./traffic.jsonl
docker compose restart kev
docker compose stop
docker compose start
docker stats
```

For CUDA include the same two Compose files when managing the deployment. Use `KEV_BIND_IP=127.0.0.1` for access only from the server or an SSH tunnel. Other containers on this Compose network can call `http://kev:8009`. Registry distribution is also supported: build with a registry-prefixed `KEV_IMAGE`, `docker push` it, then deploy with `--pull`.

## Development and verification

Controller tests do not download models or require torch:

```bash
python -m venv .venv
# Activate .venv using your shell's normal command.
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check kev_service scripts tests
python scripts/manage.py test
```

Tests cover request formats, auth, limits, lifecycle, draining, exclusivity, cancellation, restart interruption, failed reloads, publication, checkpoint compatibility, and bounded/redacted logging. The Docker `test` target installs the actual pinned upstream source and runs the same suite on Linux.

For a real 0.8B end-to-end test, download/build the production image and run:

```bash
python scripts/real-smoke.py --image kev-ranker:1.0-cpu
# NVIDIA machine, after building its CUDA image:
python scripts/real-smoke.py --image kev-ranker:1.0-cuda --cuda
```

This creates an isolated container with `--network none`, runs inference, trains/evaluates the example data, recreates the container with the persisted custom model, and checks custom inference. It removes only its own generated test container/volume afterward. CPU training may take a long time; increase `--timeout` if necessary. The ordinary smoke client can test LAN connectivity separately.

Upstream source is pinned to `65b1e763943ace53fd3a4aae00d8d08dbd849772`; all four model revisions are pinned in `kev_service/catalog.py`. The downloader preserves the checkpoint's base revision or resolves/records one if absent. Full-weight 27B downloads only the base tokenizer/config, avoiding a duplicate full base model. Model bundle hashes are checked before packaging. Core ML dependencies are pinned in `constraints.txt`; the complete installed versions are recorded at `/opt/installed-requirements.txt`. Remaining transitive dependencies and OS package updates mean rebuilds are not bit-for-bit reproducible. Preserve a tested image tag/archive for rollback.
