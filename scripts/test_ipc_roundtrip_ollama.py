"""Smoke test: IPC roundtrip to Brain Worker with Ollama enabled.

This validates the *multiprocess* path (core <-> brain worker) still works
after Ollama/model updates.

- Starts the brain worker
- Sends a TEXT request that should NOT require tools
- Sends a TEXT request that triggers a tool (fastpath)
- Confirms RESULTS are received
- Shuts down cleanly

Run:
  python scripts/test_ipc_roundtrip_ollama.py

Notes:
- The brain worker always initializes STT; using whisper_model=tiny keeps it light.
- Requires Ollama running on http://127.0.0.1:11434 and the configured model present.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wyzer.core.ipc import new_id, safe_put
from wyzer.core.process_manager import start_brain_process, stop_brain_process


def _wait_for_result(brain_to_core_q, req_id: str, timeout_sec: float) -> dict:
    deadline = time.time() + timeout_sec
    last_log = None

    while time.time() < deadline:
        try:
            msg = brain_to_core_q.get(timeout=1.0)
        except Exception:
            continue

        if msg.get("type") == "LOG":
            last_log = msg
            continue

        if msg.get("type") == "RESULT" and msg.get("id") == req_id:
            return msg

    raise TimeoutError(f"Did not receive RESULT in time (last_log={last_log})")


def main() -> int:
    config = {
        "log_level": "INFO",
        "whisper_model": "tiny",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "llm_mode": "ollama",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "llama3.2:3b",
        "llm_timeout": 60,
        "tts_enabled": False,
    }

    proc, core_to_brain_q, brain_to_core_q = start_brain_process(config)

    try:
        # 1) LLM-only response (used to fail when model returned tool='reply')
        req1 = new_id()
        safe_put(core_to_brain_q, {"type": "TEXT", "id": req1, "text": "Say hello and nothing else.", "meta": {}})
        msg1 = _wait_for_result(brain_to_core_q, req1, timeout_sec=90.0)
        reply1 = (msg1.get("reply") or "").strip()
        if not reply1 or "error" in reply1.lower() and reply1.startswith("(error:"):
            raise AssertionError(f"Unexpected reply for req1: {msg1}")

        # 2) Tool fastpath inside orchestrator (verifies tool registry works in worker)
        req2 = new_id()
        safe_put(core_to_brain_q, {"type": "TEXT", "id": req2, "text": "what time is it", "meta": {}})
        msg2 = _wait_for_result(brain_to_core_q, req2, timeout_sec=30.0)
        reply2 = (msg2.get("reply") or "").strip().lower()
        if not reply2 or "it is" not in reply2:
            raise AssertionError(f"Unexpected reply for req2: {msg2}")

        print("OK: multiprocess brain worker + ollama roundtrip")
        return 0

    finally:
        stop_brain_process(proc, core_to_brain_q)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
