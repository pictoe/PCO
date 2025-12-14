"""wyzer.core.process_manager

Small process manager for Wyzer multiprocess split.

Windows compatibility notes:
- must guard process spawning under if __name__ == "__main__"
- call multiprocessing.freeze_support() for portable/exe builds
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Tuple

from wyzer.core.ipc import safe_put


def start_brain_process(config: Dict[str, Any]) -> Tuple[mp.Process, mp.Queue, mp.Queue]:
    ctx = mp.get_context("spawn")

    core_to_brain_q: mp.Queue = ctx.Queue(maxsize=int(config.get("ipc_queue_maxsize", 50)))
    brain_to_core_q: mp.Queue = ctx.Queue(maxsize=int(config.get("ipc_queue_maxsize", 50)))

    from wyzer.core.brain_worker import run_brain_worker

    proc = ctx.Process(
        target=run_brain_worker,
        args=(core_to_brain_q, brain_to_core_q, config),
        name="WyzerBrainWorker",
        daemon=True,
    )
    proc.start()
    return proc, core_to_brain_q, brain_to_core_q


def stop_brain_process(proc: mp.Process, core_to_brain_q: mp.Queue, timeout: float = 2.0) -> None:
    try:
        safe_put(core_to_brain_q, {"type": "SHUTDOWN"}, timeout=0.25)
    except Exception:
        pass

    if proc is None:
        return

    try:
        proc.join(timeout=timeout)
    except Exception:
        return

    if proc.is_alive():
        try:
            proc.terminate()
        except Exception:
            pass
