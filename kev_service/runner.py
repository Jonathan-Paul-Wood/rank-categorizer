"""All ML code lives in killable process groups, never in the controller."""
import asyncio
import os
import signal
import sys
import time

import httpx

from .debug_log import redact


async def terminate(process):
    if process is None or process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 10)
    except TimeoutError:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        await process.wait()


class ProcessRunner:
    def __init__(self, settings, logger):
        self.settings, self.logger = settings, logger
        self.inference = None
        self.output_task = None
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:8010", trust_env=False,
                                        timeout=settings.inference_timeout)

    @property
    def alive(self):
        return self.inference is not None and self.inference.returncode is None

    def env(self):
        env = dict(os.environ)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   HF_HOME=str(self.settings.bundle_dir / "hf"),
                   OMP_NUM_THREADS=str(self.settings.cpu_threads),
                   MKL_NUM_THREADS=str(self.settings.cpu_threads), PYTHONUNBUFFERED="1",
                   KEV_DEVICE=self.settings.device, KEV_DTYPE=self.settings.dtype,
                   KEV_API_KEY="", KEV_CUDA_GRAPHS="0", KEV_FUSED="0", KEV_BACKEND="torch",
                   KEV_TRUNCATE_STATES="0", KEV_DATE_FACTS="0")
        return env

    async def _spawn(self, *args):
        return await asyncio.create_subprocess_exec(
            sys.executable, "-m", "kev_service.worker", *map(str, args), env=self.env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )

    async def _drain(self, process, job_id=None):
        # Preserve line boundaries for redaction even when a secret straddles pipe reads.
        pending = b""
        oversized = False
        def emit(raw):
            output = raw.decode("utf-8", errors="replace")
            self.logger.emit("worker.output", job_id=job_id, output=output)
            print(redact(output, (self.settings.api_key,)), file=sys.stderr, end="", flush=True)

        while chunk := await process.stdout.read(8192):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if oversized or len(line) > 8192:
                    emit(b"[oversized worker output line omitted]\n")
                else:
                    emit(line + b"\n")
                oversized = False
            if len(pending) > 8192:
                oversized = True
                pending = b""
        if oversized:
            emit(b"[oversized worker output line omitted]\n")
        elif pending:
            emit(pending)

    async def start(self, path):
        await self.stop()
        self.inference = await self._spawn("serve", path, self.settings.device)
        self.output_task = asyncio.create_task(self._drain(self.inference))
        deadline = time.monotonic() + self.settings.startup_timeout
        try:
            while time.monotonic() < deadline:
                if not self.alive:
                    raise RuntimeError("inference process exited during startup")
                try:
                    response = await self.client.get("/v1/models", timeout=1)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
            raise TimeoutError("model startup timed out")
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        await terminate(self.inference)
        if self.output_task:
            await self.output_task
        self.inference = None
        self.output_task = None

    async def request(self, body):
        response = await self.client.post("/v1/systemone", json=body)
        return response.status_code, response.json()

    async def details(self):
        response = await self.client.get("/v1/models", timeout=5)
        response.raise_for_status()
        return response.json()["models"][0]

    async def run(self, operation, *args, job_id=None, timeout=None):
        process = await self._spawn(operation, *args)
        drain = asyncio.create_task(self._drain(process, job_id))
        try:
            await asyncio.wait_for(process.wait(), timeout or self.settings.training_timeout)
            if process.returncode:
                raise RuntimeError(f"{operation} worker exited with status {process.returncode}; inspect debug/Docker logs")
        finally:
            await terminate(process)
            await drain

    async def close(self):
        await self.stop()
        await self.client.aclose()

