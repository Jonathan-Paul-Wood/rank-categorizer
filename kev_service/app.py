import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .catalog import ALIASES
from .config import Settings
from .debug_log import DebugLog
from .runner import ProcessRunner
from .schema import DecisionRequest, TrainOptions, strict_json, validate_dataset
from .service import Service
from .storage import Registry


class TrafficMiddleware:
    """Authenticate before reading input and cap streamed bodies, including chunked multipart."""
    def __init__(self, app, settings, logger):
        self.app, self.settings, self.logger = app, settings, logger

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        headers = dict(scope["headers"])
        incoming_id = headers.get(b"x-typesafe-request-id", b"").decode("ascii", errors="ignore")
        request_id = incoming_id if 0 < len(incoming_id) <= 128 and incoming_id.isprintable() else uuid.uuid4().hex
        path = scope["path"]
        scope.setdefault("state", {})["request_id"] = request_id
        self.logger.emit("api.request", request_id=request_id, method=scope["method"], path=path,
                         headers={k.decode("latin1"): v.decode("latin1") for k, v in headers.items()})
        status = 500
        response_body = bytearray()

        async def wrapped_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message["headers"]) + [
                    (b"x-typesafe-request-id", request_id.encode()), (b"x-request-id", request_id.encode())]
            elif message["type"] == "http.response.body" and self.logger.enabled:
                if len(response_body) < 1024 * 1024:
                    response_body.extend(message.get("body", b"")[:1024 * 1024 - len(response_body)])
            await send(message)

        async def reject(code, message):
            await JSONResponse({"detail": message}, code)(scope, receive, wrapped_send)

        try:
            if path.startswith("/v1") and self.settings.api_key:
                expected = f"Bearer {self.settings.api_key}".encode()
                if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                    await reject(401, "missing or invalid bearer API key")
                    return
            limit = (self.settings.max_upload_bytes + 65536 if path == "/v1/fine-tunes"
                     else self.settings.max_request_bytes)
            try:
                declared = int(headers.get(b"content-length", b"0"))
            except ValueError:
                await reject(400, "invalid Content-Length")
                return
            if declared < 0 or declared > limit:
                await reject(413, "request exceeds configured size limit")
                return
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > limit:
                    await reject(413, "request exceeds configured size limit")
                    return
                if not message.get("more_body", False):
                    break
            if self.logger.enabled and body:
                if headers.get(b"content-type", b"").startswith(b"application/json"):
                    try:
                        captured = strict_json(body)
                    except (ValueError, UnicodeError, RecursionError):
                        captured = {"omitted": "invalid_json", "byte_count": len(body)}
                else:
                    captured = {"omitted": "multipart_or_binary", "byte_count": len(body)}
                self.logger.emit("api.input", request_id=request_id, body=captured)
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, wrapped_send)
        finally:
            fields = {}
            if response_body:
                try:
                    fields["body"] = json.loads(response_body)
                except (ValueError, UnicodeError):
                    fields["body"] = {"omitted": "oversized_response"}
            self.logger.emit("api.response", request_id=request_id, status=status,
                             duration_ms=round((time.monotonic() - started) * 1000, 3), **fields)


def create_app(settings=None, runner_factory=ProcessRunner):
    settings = settings or Settings.from_env()
    logger = DebugLog(settings.debug_dir, settings.debug_mode, (settings.api_key,))

    @asynccontextmanager
    async def lifespan(app):
        try:
            registry = Registry(settings)
            service = Service(settings, registry, runner_factory(settings, logger), logger)
            app.state.service = service
            await service.start()
            try:
                yield
            finally:
                await service.close()
        finally:
            logger.close()

    app = FastAPI(title="Kev Ranker", version="1.0.0", lifespan=lifespan)
    app.add_middleware(TrafficMiddleware, settings=settings, logger=logger)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo uploaded records, credentials, or non-finite values through validation errors.
        return JSONResponse({"detail": [{"loc": list(error["loc"]), "msg": error["msg"],
                                          "type": error["type"]} for error in exc.errors()]}, 422)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        service = app.state.service
        return JSONResponse({"status": service.state, "model": settings.model_id, "error": service.error},
                            200 if service.state == "ready" and service.runner.alive else 503)

    @app.post("/v1/systemone")
    async def systemone(request: Request, validated: DecisionRequest):
        try:
            body = DecisionRequest.model_validate(strict_json(await request.body()))
        except (ValueError, UnicodeError, RecursionError):
            raise HTTPException(422, "invalid Kev request; see /docs") from None
        if len(body.questions) > settings.max_questions:
            raise HTTPException(422, "too many questions")
        status, output = await app.state.service.answer(body, request.state.request_id)
        return JSONResponse(output, status)

    @app.get("/v1/models")
    async def models():
        service = app.state.service
        if service.state != "ready":
            raise HTTPException(503, f"model unavailable: {service.state}")
        try:
            details = await service.runner.details()
        except Exception:
            service.schedule_recovery()
            raise HTTPException(503, "model worker unavailable") from None
        details.pop("name", None)
        return {"models": [{**details, "name": name, "id": name, "active_model": settings.model_id,
                            "calibrated": not settings.model_id.startswith("ft-")}
                           for name in (settings.model_id, *ALIASES)]}

    @app.get("/v1/model-catalog")
    async def catalog():
        return {"models": app.state.service.registry.catalog()}

    @app.post("/v1/fine-tunes", status_code=202)
    async def submit(request: Request, name: str = Form(min_length=1, max_length=100),
                     options: str = Form(default="{}", max_length=4096), data: UploadFile = File()):
        try:
            options_obj = TrainOptions.model_validate(strict_json(options))
            raw = await data.read(settings.max_upload_bytes + 1)
            if len(raw) > settings.max_upload_bytes:
                raise HTTPException(413, "dataset exceeds KEV_MAX_UPLOAD_BYTES")
            records = validate_dataset(raw, settings.max_records, settings.max_questions)
            logger.emit("training.input", name=name, filename=data.filename, byte_count=len(raw),
                        record_count=len(records), options=options_obj.model_dump(),
                        request_id=request.state.request_id)
            return await app.state.service.submit(name, options_obj, records, request.state.request_id)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HTTPException(422, str(exc)) from None
        except (RuntimeError, TimeoutError):
            raise HTTPException(422, "dataset/tokenizer preflight failed; inspect debug/Docker logs") from None
        finally:
            await data.close()

    @app.get("/v1/fine-tunes")
    async def jobs():
        return {"jobs": app.state.service.jobs()}

    @app.get("/v1/fine-tunes/{job_id}")
    async def job(job_id: str):
        service = app.state.service
        output = service.job(job_id)
        report = service.registry.models / output["model_id"] / "evaluation" / "report.json"
        if report.is_file():
            output["evaluation"] = json.loads(report.read_text(encoding="utf-8"))
        return output

    @app.post("/v1/fine-tunes/{job_id}/cancel", status_code=202)
    async def cancel(job_id: str):
        return await app.state.service.cancel(job_id)

    return app

