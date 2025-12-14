"""wyzer.core.ipc

Single source of truth for Wyzer multiprocess IPC message schemas.

Communication is strictly via multiprocessing.Queue:
- core_to_brain_q
- brain_to_core_q

Messages are plain dicts to keep pickling/simple-queue compatibility.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, TypedDict, Literal


JsonDict = Dict[str, Any]


class AudioRequest(TypedDict, total=False):
    type: Literal["AUDIO"]
    id: str
    wav_path: Optional[str]
    pcm_bytes: Optional[bytes]
    sample_rate: int
    meta: JsonDict


class TextRequest(TypedDict, total=False):
    type: Literal["TEXT"]
    id: str
    text: str
    meta: JsonDict


class Interrupt(TypedDict):
    type: Literal["INTERRUPT"]
    reason: Literal["hotword"]


class Shutdown(TypedDict):
    type: Literal["SHUTDOWN"]


class BrainResult(TypedDict, total=False):
    type: Literal["RESULT"]
    id: str
    reply: str
    tool_calls: Optional[list]
    tts_text: Optional[str]
    meta: JsonDict


class LogEvent(TypedDict, total=False):
    type: Literal["LOG"]
    level: str
    msg: str
    meta: JsonDict


def new_id() -> str:
    return uuid.uuid4().hex


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_put(q, msg: Dict[str, Any], timeout: float = 0.25) -> bool:
    """Best-effort Queue.put with timeout.

    Returns False on failure (non-fatal).
    """
    try:
        q.put(msg, timeout=timeout)
        return True
    except Exception:
        return False
