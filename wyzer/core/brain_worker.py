"""wyzer.core.brain_worker

Brain Worker process:
- STT (Whisper)
- Orchestrator (LLM + tools)
- TTS

Receives requests from core via core_to_brain_q and sends results/logs via brain_to_core_q.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from wyzer.core.config import Config
from wyzer.core.ipc import now_ms, safe_put
from wyzer.core.logger import get_logger, init_logger
from wyzer.stt.stt_router import STTRouter
from wyzer.tts.tts_router import TTSRouter


def _apply_config(config_dict: Dict[str, Any]) -> None:
    # Logger first
    log_level = str(config_dict.get("log_level", "INFO")).upper()
    init_logger(log_level)

    # Ensure orchestrator uses the worker's config (it reads Config.*)
    if "ollama_url" in config_dict:
        Config.OLLAMA_BASE_URL = str(config_dict["ollama_url"])
    if "ollama_model" in config_dict:
        Config.OLLAMA_MODEL = str(config_dict["ollama_model"])
    if "llm_timeout" in config_dict:
        Config.LLM_TIMEOUT = int(config_dict["llm_timeout"])

    # Whisper defaults
    if "whisper_model" in config_dict:
        Config.WHISPER_MODEL = str(config_dict["whisper_model"])
    if "whisper_device" in config_dict:
        Config.WHISPER_DEVICE = str(config_dict["whisper_device"])
    if "whisper_compute_type" in config_dict:
        Config.WHISPER_COMPUTE_TYPE = str(config_dict["whisper_compute_type"])


def _read_wav_to_float32(wav_path: str) -> np.ndarray:
    import wave

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got channels={channels}")
    if sample_rate != Config.SAMPLE_RATE:
        # Keep it strict for now; core should record at Config.SAMPLE_RATE.
        raise ValueError(f"Unexpected sample_rate={sample_rate}, expected {Config.SAMPLE_RATE}")

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio


class _TTSController:
    def __init__(self, tts: Optional[TTSRouter], brain_to_core_q, simulate: bool = False):
        self._tts = tts
        self._simulate = simulate
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="BrainTTS", daemon=True)
        self._thread.start()
        self._brain_to_core_q = brain_to_core_q

    def enqueue(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return
        self._queue.put({"text": text, "meta": meta or {}})

    def interrupt(self) -> None:
        with self._lock:
            self._stop_event.set()
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

    def clear_stop(self) -> None:
        with self._lock:
            self._stop_event.clear()

    def shutdown(self) -> None:
        self._running = False
        self.interrupt()
        self._queue.put({"text": "", "meta": {"_shutdown": True}})
        self._thread.join(timeout=1.0)

    def _simulate_speak(self, duration_sec: float) -> bool:
        end = time.time() + max(0.0, duration_sec)
        while time.time() < end:
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return True

    def _loop(self) -> None:
        logger = get_logger()
        while self._running:
            item = self._queue.get()
            meta = item.get("meta") or {}
            if meta.get("_shutdown"):
                return

            text = (item.get("text") or "").strip()
            if not text:
                continue

            safe_put(self._brain_to_core_q, {"type": "LOG", "level": "DEBUG", "msg": "tts_started"})

            ok = False
            try:
                if self._simulate or bool(meta.get("simulate_tts_sec")):
                    ok = self._simulate_speak(float(meta.get("simulate_tts_sec", 2.0)))
                elif self._tts:
                    self.clear_stop()
                    ok = self._tts.speak(text, self._stop_event)
                else:
                    ok = False
            except Exception as e:
                logger.error(f"TTS error: {e}")
                ok = False
            finally:
                safe_put(
                    self._brain_to_core_q,
                    {
                        "type": "LOG",
                        "level": "DEBUG",
                        "msg": "tts_finished" if ok else "tts_interrupted",
                    },
                )


def run_brain_worker(core_to_brain_q, brain_to_core_q, config_dict: Dict[str, Any]) -> None:
    """Entrypoint for the Brain Worker process."""

    _apply_config(config_dict)
    logger = get_logger()

    # Init heavy components once
    stt = STTRouter(
        whisper_model=str(config_dict.get("whisper_model", Config.WHISPER_MODEL)),
        whisper_device=str(config_dict.get("whisper_device", Config.WHISPER_DEVICE)),
        whisper_compute_type=str(config_dict.get("whisper_compute_type", Config.WHISPER_COMPUTE_TYPE)),
    )

    llm_mode = str(config_dict.get("llm_mode", Config.LLM_MODE))
    if llm_mode != "ollama":
        logger.info("LLM disabled in brain worker")

    tts_enabled = bool(config_dict.get("tts_enabled", True))
    tts_router: Optional[TTSRouter] = None
    if tts_enabled:
        try:
            tts_router = TTSRouter(
                engine=str(config_dict.get("tts_engine", "piper")),
                piper_exe_path=str(config_dict.get("piper_exe_path", "./wyzer/assets/piper/piper.exe")),
                piper_model_path=str(config_dict.get("piper_model_path", "./wyzer/assets/piper/en_US-lessac-medium.onnx")),
                piper_speaker_id=config_dict.get("piper_speaker_id"),
                output_device=config_dict.get("tts_output_device"),
                enabled=True,
            )
        except Exception as e:
            logger.error(f"Failed to init TTS: {e}")
            tts_router = None

    simulate_tts = bool(config_dict.get("simulate_tts", False))
    tts_controller = _TTSController(tts_router, brain_to_core_q, simulate=simulate_tts)

    interrupt_generation = 0

    safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "brain_worker_started"})

    while True:
        msg = core_to_brain_q.get()
        mtype = (msg or {}).get("type")

        if mtype == "SHUTDOWN":
            safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "brain_worker_shutdown"})
            try:
                tts_controller.shutdown()
            except Exception:
                pass
            return

        if mtype == "INTERRUPT":
            interrupt_generation += 1
            tts_controller.interrupt()
            safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "interrupt_ack"})
            continue

        if mtype not in {"AUDIO", "TEXT"}:
            safe_put(
                brain_to_core_q,
                {"type": "LOG", "level": "WARNING", "msg": f"unknown_msg_type:{mtype}"},
            )
            continue

        req_id = msg.get("id") or ""
        request_gen = interrupt_generation
        start_ms = now_ms()

        try:
            user_text: str = ""
            stt_ms = 0

            if mtype == "AUDIO":
                stt_start = now_ms()

                wav_path = msg.get("wav_path")
                pcm_bytes = msg.get("pcm_bytes")

                if wav_path:
                    audio = _read_wav_to_float32(wav_path)
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass
                elif pcm_bytes:
                    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
                else:
                    audio = np.array([], dtype=np.float32)

                user_text = stt.transcribe(audio)
                stt_ms = now_ms() - stt_start

                if not user_text:
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": "(I didn't catch that.)",
                            "tool_calls": None,
                            "tts_text": None,
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": 0,
                                    "tool_ms": 0,
                                    "tts_start_ms": None,
                                    "total_ms": now_ms() - start_ms,
                                }
                            },
                        },
                    )
                    continue

            else:
                user_text = str(msg.get("text") or "")

            # LLM + tools via orchestrator
            llm_start = now_ms()
            from wyzer.core.orchestrator import handle_user_text

            result_dict = handle_user_text(user_text) if llm_mode == "ollama" else {"reply": user_text}
            llm_ms = now_ms() - llm_start

            reply = (result_dict or {}).get("reply", "")
            exec_summary = (result_dict or {}).get("execution_summary")
            tool_calls = None
            tool_ms = 0
            if exec_summary and isinstance(exec_summary, dict):
                tool_calls = exec_summary.get("ran")
                try:
                    for r in tool_calls or []:
                        res = (r or {}).get("result")
                        if isinstance(res, dict) and isinstance(res.get("latency_ms"), int):
                            tool_ms += int(res.get("latency_ms") or 0)
                except Exception:
                    tool_ms = 0

            tts_text: Optional[str] = reply
            tts_start_ms: Optional[int] = None

            tts_interrupted = False

            # If user interrupted while we were processing, do not speak stale reply.
            if request_gen != interrupt_generation:
                tts_text = None
                tts_interrupted = True
            elif not tts_enabled or not tts_router:
                tts_text = None

            # Enqueue speech (non-blocking)
            if tts_text:
                tts_start_ms = now_ms()
                tts_meta = msg.get("meta") or {}
                tts_controller.enqueue(tts_text, meta=tts_meta)

            total_ms = now_ms() - start_ms

            safe_put(
                brain_to_core_q,
                {
                    "type": "RESULT",
                    "id": req_id,
                    "reply": reply,
                    "tool_calls": tool_calls,
                    "tts_text": tts_text,
                    "meta": {
                        "timings": {
                            "stt_ms": stt_ms,
                            "llm_ms": llm_ms,
                            "tool_ms": tool_ms,
                            "tts_start_ms": tts_start_ms,
                            "total_ms": total_ms,
                        },
                        "tts_interrupted": tts_interrupted,
                        "user_text": user_text,
                    },
                },
            )

        except Exception as e:
            err = str(e)
            safe_put(brain_to_core_q, {"type": "LOG", "level": "ERROR", "msg": f"brain_worker_error:{err}", "meta": {"trace": traceback.format_exc()}})
            safe_put(
                brain_to_core_q,
                {
                    "type": "RESULT",
                    "id": req_id,
                    "reply": f"(error: {err})",
                    "tool_calls": None,
                    "tts_text": None,
                    "meta": {"timings": {"total_ms": now_ms() - start_ms}, "error": True},
                },
            )
