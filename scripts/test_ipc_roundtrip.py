"""Smoke test: IPC roundtrip to Brain Worker.

- Starts the brain worker
- Sends a TEXT request
- Confirms a RESULT is received within timeout
- Shuts down cleanly

Run:
  python scripts/test_ipc_roundtrip.py
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


def main() -> int:
    config = {
        "log_level": "INFO",
        "whisper_model": "tiny",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "llm_mode": "off",  # keep test deterministic/fast
        "tts_enabled": False,
    }

    proc, core_to_brain_q, brain_to_core_q = start_brain_process(config)
    req_id = new_id()

    safe_put(core_to_brain_q, {"type": "TEXT", "id": req_id, "text": "say hello", "meta": {}})

    # First run may spend time loading Whisper model; be generous.
    deadline = time.time() + 60.0
    got_result = False
    while time.time() < deadline:
        try:
            msg = brain_to_core_q.get(timeout=1.0)
        except Exception:
            continue
        if msg.get("type") == "LOG":
            continue
        if msg.get("type") == "RESULT" and msg.get("id") == req_id:
            got_result = True
            break

    stop_brain_process(proc, core_to_brain_q)

    if not got_result:
        raise SystemExit("Did not receive RESULT in time")

    print("OK: received RESULT")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
