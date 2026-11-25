import asyncio
import json
import shutil
import uuid

from fastapi import HTTPException

from .catalog import ALIASES
from .debug_log import now
from .schema import split_dataset
from .storage import write_json

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Service:
    def __init__(self, settings, registry, runner, logger):
        self.settings, self.registry, self.runner, self.logger = settings, registry, runner, logger
        self.state, self.error = "loading", None
        self.condition = asyncio.Condition()
        self.submission = asyncio.Lock()
        self.inflight = 0
        self.active_job = None
        self.job_task = None
        self.model_task = None
        self.watch_task = None
        self.closing = False
        self.job_started = asyncio.Event()

    def save(self, job):
        job["updated_at"] = now()
        write_json(self.registry.jobs / job["id"] / "job.json", job)

    def job(self, job_id):
        # IDs never become caller-controlled paths.
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(404, "job not found")
        try:
            return json.loads((self.registry.jobs / job_id / "job.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "job not found") from None

    def jobs(self):
        return sorted((json.loads(p.read_text(encoding="utf-8")) for p in self.registry.jobs.glob("*/job.json")),
                      key=lambda item: item["created_at"], reverse=True)

    async def start(self):
        for job in self.jobs():
            if job["status"] not in TERMINAL:
                # Atomic model publication can precede the final job write by one crash window.
                published = self.registry.models / job["model_id"] / "model.json"
                if published.exists():
                    job["status"] = "succeeded"
                else:
                    job.update(status="interrupted", error="controller restarted before completion")
                self.save(job)
        self.model_task = asyncio.create_task(self.load())
        self.watch_task = asyncio.create_task(self.watch())

    async def drained(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.inflight == 0)

    async def load(self):
        self.state, self.error = "loading", None
        self.logger.emit("model.loading", model=self.settings.model_id)
        try:
            await self.runner.start(self.registry.active_path)
            self.state = "ready"
            self.logger.emit("model.ready", model=self.settings.model_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state, self.error = "failed", str(exc)
            self.logger.emit("model.error", error=str(exc))

    async def recover(self):
        await self.drained()
        await self.load()

    def schedule_recovery(self):
        if self.state == "ready":
            self.state = "recovering"
            self.model_task = asyncio.create_task(self.recover())

    async def watch(self):
        while True:
            await asyncio.sleep(1)
            if self.state == "ready" and not self.runner.alive:
                self.schedule_recovery()

    async def answer(self, request, request_id=None):
        if request.model not in (*ALIASES, self.settings.model_id):
            raise HTTPException(404, "requested model is not active; select KEV_MODEL and recreate the container")
        async with self.condition:
            if self.state != "ready":
                raise HTTPException(503, f"inference unavailable: {self.state}", headers={"Retry-After": "5"})
            if self.inflight >= self.settings.max_concurrency:
                raise HTTPException(429, "inference concurrency limit reached", headers={"Retry-After": "1"})
            self.inflight += 1
        try:
            return await self.runner.request(request.model_dump())
        except Exception as exc:
            self.schedule_recovery()
            self.logger.emit("inference.error", request_id=request_id, error=str(exc))
            raise HTTPException(503, "inference worker failed; recovering", headers={"Retry-After": "5"}) from None
        finally:
            async with self.condition:
                self.inflight -= 1
                self.condition.notify_all()

    async def submit(self, name, options, records, request_id=None):
        async with self.submission:
            if self.active_job:
                raise HTTPException(409, "a fine-tuning job is already active")
            if self.state != "ready":
                raise HTTPException(503, f"cannot start fine-tuning while {self.state}")
            if self.registry.bundled_id == "kev-27b":
                raise HTTPException(422, "Kev-27B is inference-only")
            train, evaluation = split_dataset(records, options.seed)
            job_id = uuid.uuid4().hex
            directory = self.registry.jobs / job_id
            directory.mkdir(mode=0o700)
            try:
                for filename, items in (("input.jsonl", records), ("train.jsonl", train),
                                        ("evaluation.jsonl", evaluation)):
                    (directory / filename).write_text(
                        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in items),
                        encoding="utf-8")
                # Tokenizer/context validation runs before admission closes and without loading a second model.
                await self.runner.run("validate", self.settings.bundle_dir / "checkpoint", directory / "input.jsonl",
                                      timeout=self.settings.validation_timeout, job_id=job_id)
            except BaseException:
                shutil.rmtree(directory)
                raise
            if self.state != "ready":
                shutil.rmtree(directory)
                raise HTTPException(503, "model became unavailable during validation")
            job = {"id": job_id, "name": name, "model_id": f"ft-{job_id}", "status": "draining",
                   "parent_model": self.registry.bundled_id, "options": options.model_dump(),
                   "train_records": len(train), "evaluation_records": len(evaluation), "created_at": now(),
                   "status_url": f"/v1/fine-tunes/{job_id}", "error": None}
            self.save(job)
            self.logger.emit("training.accepted", request_id=request_id, job_id=job_id)
            # No await between closing admission and reserving the single training slot.
            self.active_job = job_id
            self.state = "draining"
            self.job_started.clear()
            self.job_task = asyncio.create_task(self.run_job(job))
            return job

    async def run_job(self, job):
        self.job_started.set()
        directory = self.registry.jobs / job["id"]
        try:
            await self.drained()
            await self.runner.stop()
            self.state = "training"
            job["status"] = "training"
            self.save(job)
            self.logger.emit("training.started", job_id=job["id"], options=job["options"])
            await self.runner.run("train", self.settings.bundle_dir / "checkpoint", directory,
                                  self.settings.device, job_id=job["id"])
            job["status"] = "evaluating"
            self.save(job)
            self.state = "evaluating"
            await self.runner.run("evaluate", directory / "staging" / "checkpoint", directory / "evaluation.jsonl",
                                  directory / "staging" / "evaluation", self.settings.device, job_id=job["id"])
            if not (directory / "staging" / "checkpoint" / "head.pt").is_file():
                raise ValueError("training produced no complete checkpoint")
            metadata = {"id": job["model_id"], "name": job["name"], "parent_model": job["parent_model"],
                        "parent_revision": self.registry.bundle["revision"],
                        "base_revision": self.registry.bundle["base_revision"], "job_id": job["id"],
                        "created_at": now(), "calibrated": False, "temperature": 1.0}
            write_json(directory / "staging" / "model.json", metadata)
            # Same filesystem: readers see the entire checkpoint/report together, or nothing.
            (directory / "staging").rename(self.registry.models / job["model_id"])
            job["status"] = "succeeded"
        except asyncio.CancelledError:
            job.update(status="interrupted" if self.closing else "cancelled", error=None)
        except Exception as exc:
            job.update(status="failed", error=str(exc))
            self.logger.emit("training.error", job_id=job["id"], error=str(exc))
        finally:
            try:
                self.save(job)
                self.logger.emit("training.finished", job_id=job["id"], status=job["status"])
            except OSError as exc:
                self.logger.emit("training.persistence_error", job_id=job["id"], error=str(exc))
            finally:
                try:
                    if not self.closing:
                        # Cancellation during draining must still wait for already admitted requests.
                        self.state = "recovering"
                        await self.drained()
                        await self.load()
                        if self.error:
                            job["reload_error"] = self.error
                            self.save(job)
                finally:
                    self.active_job = None

    async def cancel(self, job_id):
        job = self.job(job_id)
        if job["status"] in TERMINAL:
            return job
        if job_id == self.active_job and self.job_task:
            # Do not await model reload in the HTTP cancellation request.
            if job["status"] != "cancelling":
                job["status"] = "cancelling"
                self.save(job)
                await self.job_started.wait()
                self.job_task.cancel()
            return job
        raise HTTPException(409, "job is not cancellable")

    async def close(self):
        self.closing = True
        self.state = "stopping"
        for task in (self.watch_task, self.model_task, self.job_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.runner.close()

