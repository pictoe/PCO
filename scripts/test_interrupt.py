"""Smoke test: Brain Worker interrupt.

This test uses simulated TTS (no audio device required):
- Start brain worker with simulate_tts=True
- Send TEXT request with meta.simulate_tts_sec (long)
- Immediately send INTERRUPT
- Confirm we see either:
  - a LOG msg 'interrupt_ack', OR
  - a LOG msg 'tts_interrupted', OR
  - a RESULT with meta.tts_interrupted=True

Run:
  python scripts/test_interrupt.py
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
        "log_level": "DEBUG",
        "whisper_model": "tiny",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "llm_mode": "off",  # keep deterministic
        "tts_enabled": True,
        "simulate_tts": True,
    }

    proc, core_to_brain_q, brain_to_core_q = start_brain_process(config)

    req_id = new_id()
    safe_put(
        core_to_brain_q,
        {
            "type": "TEXT",
            "id": req_id,
            "text": "this should take a while",
            "meta": {"simulate_tts_sec": 5.0},
        },
    )

    # Give it a tiny moment to start TTS thread
    time.sleep(0.2)

    safe_put(core_to_brain_q, {"type": "INTERRUPT", "reason": "hotword"})

    # First run may spend time loading Whisper model; be generous.
    deadline = time.time() + 60.0
    ack = False

    while time.time() < deadline:
        try:
            msg = brain_to_core_q.get(timeout=1.0)
        except Exception:
            continue
        if msg.get("type") == "LOG":
            text = msg.get("msg")
            if text in {"interrupt_ack", "tts_interrupted"}:
                ack = True
                break
        if msg.get("type") == "RESULT" and msg.get("id") == req_id:
            meta = msg.get("meta") or {}
            if meta.get("tts_interrupted") is True:
                ack = True
                break

    stop_brain_process(proc, core_to_brain_q)

    if not ack:
        raise SystemExit("Did not observe interrupt acknowledgement")

    print("OK: interrupt acknowledged")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
