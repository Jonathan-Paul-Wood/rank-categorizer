import asyncio
import os
import sys

import pytest

from kev_service.debug_log import DebugLog
from kev_service.runner import ProcessRunner


@pytest.mark.parametrize("cancel", [False, True])
async def test_real_subprocess_is_terminated(settings, monkeypatch, cancel):
    log = DebugLog(settings.debug_dir)
    runner = ProcessRunner(settings, log)
    process = None
    spawned = asyncio.Event()

    async def spawn(*args):
        nonlocal process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; print('started',flush=True); time.sleep(60)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix")
        spawned.set()
        return process

    monkeypatch.setattr(runner, "_spawn", spawn)
    try:
        task = asyncio.create_task(runner.run("train", timeout=0.1 if not cancel else 60))
        await spawned.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert process.returncode is not None
    finally:
        await runner.close()
        log.close()

