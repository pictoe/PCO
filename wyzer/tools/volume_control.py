"""Volume control tool.

Provides true master volume get/set/change and per-application (per-process) volume
control on Windows using pycaw.

This complements the legacy media-key based volume tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wyzer.tools.tool_base import ToolBase


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _normalize_name(name: str) -> str:
    s = (name or "").strip().lower()
    if s.endswith(".exe"):
        s = s[: -len(".exe")]
    # Keep alnum and spaces only
    cleaned = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
    return " ".join("".join(cleaned).split())


def _score_candidate(query_norm: str, candidate_norm: str) -> float:
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if candidate_norm.startswith(query_norm):
        return 0.92
    if query_norm.startswith(candidate_norm):
        return 0.88
    if query_norm in candidate_norm:
        return 0.75

    q_tokens = set(query_norm.split())
    c_tokens = set(candidate_norm.split())
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    return 0.55 * overlap


@dataclass(frozen=True)
class _SessionInfo:
    process_name: str  # e.g., "chrome.exe"
    display_name: str


class VolumeControlTool(ToolBase):
    """True master + per-application volume control via pycaw."""

    def __init__(self):
        super().__init__()
        self._name = "volume_control"
        self._description = (
            "Get/set/change system master volume and per-app volume (Windows). "
            "Supports mute/unmute/toggle and fuzzy process matching."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["master", "app"],
                    "description": "Target scope: master system volume or an app/process session.",
                },
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "change", "mute", "unmute", "toggle_mute"],
                    "description": "Action to perform.",
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Target volume level percent for action=set.",
                },
                "delta": {
                    "type": "integer",
                    "minimum": -100,
                    "maximum": 100,
                    "description": "Volume delta percent for action=change (positive or negative).",
                },
                "process": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Process/app name for scope=app (fuzzy matched, e.g., 'spotify', 'chrome').",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()

        scope = str(kwargs.get("scope") or "master").strip().lower()
        action = str(kwargs.get("action") or "").strip().lower()
        level = kwargs.get("level")
        delta = kwargs.get("delta")
        process = (kwargs.get("process") or "").strip()

        if scope not in {"master", "app"}:
            scope = "master"

        if scope == "app" and not process:
            return {
                "error": {"type": "invalid_args", "message": "scope=app requires process"},
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }

        try:
            if scope == "master":
                result = self._run_master(action=action, level=level, delta=delta)
            else:
                result = self._run_app(action=action, level=level, delta=delta, process=process)

            result["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return result

        except ImportError as e:
            return {
                "error": {
                    "type": "missing_dependency",
                    "message": f"Missing dependency for volume control: {e}",
                },
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
        except Exception as e:
            return {
                "error": {"type": "execution_error", "message": str(e)},
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }

    def _run_master(self, *, action: str, level: Any, delta: Any) -> Dict[str, Any]:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()

        # pycaw may return either an IMMDevice COM object (with Activate)
        # or an AudioDevice wrapper (pycaw.utils.AudioDevice) that exposes the
        # underlying COM device as ._dev.
        com_device = devices
        if not hasattr(com_device, "Activate") and hasattr(devices, "_dev"):
            com_device = getattr(devices, "_dev")

        if not hasattr(com_device, "Activate"):
            raise RuntimeError(
                "Unsupported pycaw device object (missing Activate). "
                "Try upgrading pycaw to a newer version."
            )

        interface = com_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        endpoint = cast(interface, POINTER(IAudioEndpointVolume))

        old_scalar = float(endpoint.GetMasterVolumeLevelScalar())
        old_level = int(round(old_scalar * 100))
        old_muted = bool(endpoint.GetMute())

        if action == "get":
            return {
                "status": "ok",
                "scope": "master",
                "level": old_level,
                "muted": old_muted,
            }

        if action == "set":
            if level is None:
                return {"error": {"type": "invalid_args", "message": "action=set requires level"}}
            new_level = _clamp_int(int(level), 0, 100)
            endpoint.SetMasterVolumeLevelScalar(new_level / 100.0, None)
            new_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            return {
                "status": "ok",
                "scope": "master",
                "old_level": old_level,
                "new_level": int(round(new_scalar * 100)),
                "muted": bool(endpoint.GetMute()),
            }

        if action == "change":
            if delta is None:
                return {"error": {"type": "invalid_args", "message": "action=change requires delta"}}
            d = _clamp_int(int(delta), -100, 100)
            new_level = _clamp_int(old_level + d, 0, 100)
            endpoint.SetMasterVolumeLevelScalar(new_level / 100.0, None)
            new_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            return {
                "status": "ok",
                "scope": "master",
                "delta": d,
                "old_level": old_level,
                "new_level": int(round(new_scalar * 100)),
                "muted": bool(endpoint.GetMute()),
            }

        if action == "mute":
            endpoint.SetMute(1, None)
            return {
                "status": "ok",
                "scope": "master",
                "level": int(round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)),
                "muted": True,
            }

        if action == "unmute":
            endpoint.SetMute(0, None)
            return {
                "status": "ok",
                "scope": "master",
                "level": int(round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)),
                "muted": False,
            }

        if action == "toggle_mute":
            endpoint.SetMute(0 if old_muted else 1, None)
            return {
                "status": "ok",
                "scope": "master",
                "level": int(round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)),
                "muted": bool(endpoint.GetMute()),
            }

        return {"error": {"type": "invalid_args", "message": f"Unknown action: {action}"}}

    def _list_sessions(self) -> List[Tuple[Any, _SessionInfo]]:
        from pycaw.pycaw import AudioUtilities

        sessions = []
        for s in AudioUtilities.GetAllSessions():
            p = getattr(s, "Process", None)
            if p is None:
                continue
            try:
                pname = p.name() or ""
            except Exception:
                pname = ""

            if not pname:
                continue

            display = getattr(s, "DisplayName", None) or ""
            sessions.append((s, _SessionInfo(process_name=pname, display_name=str(display))))
        return sessions

    def _match_session(self, process: str) -> Tuple[Optional[Any], Optional[_SessionInfo], List[Dict[str, Any]]]:
        query_norm = _normalize_name(process)
        sessions = self._list_sessions()

        scored: List[Tuple[float, Any, _SessionInfo]] = []
        for s, info in sessions:
            cand_norm = _normalize_name(info.process_name)
            score = _score_candidate(query_norm, cand_norm)
            if score > 0:
                scored.append((score, s, info))

        scored.sort(key=lambda t: t[0], reverse=True)

        candidates: List[Dict[str, Any]] = []
        for score, _s, info in scored[:5]:
            candidates.append({
                "process": info.process_name,
                "display": info.display_name,
                "score": round(float(score), 3),
            })

        if not scored:
            return None, None, candidates

        best_score, best_session, best_info = scored[0]
        # Require a reasonable match to avoid changing the wrong app.
        if best_score < 0.6:
            return None, None, candidates

        return best_session, best_info, candidates

    def _run_app(self, *, action: str, level: Any, delta: Any, process: str) -> Dict[str, Any]:
        session, info, candidates = self._match_session(process)
        if session is None or info is None:
            return {
                "error": {
                    "type": "app_not_found",
                    "message": f"No active audio session matched '{process}'",
                },
                "scope": "app",
                "requested_process": process,
                "candidates": candidates,
            }

        simple = session.SimpleAudioVolume
        old_scalar = float(simple.GetMasterVolume())
        old_level = int(round(old_scalar * 100))
        old_muted = bool(simple.GetMute())

        if action == "get":
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "level": old_level,
                "muted": old_muted,
                "candidates": candidates,
            }

        if action == "set":
            if level is None:
                return {"error": {"type": "invalid_args", "message": "action=set requires level"}}
            new_level = _clamp_int(int(level), 0, 100)
            simple.SetMasterVolume(new_level / 100.0, None)
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "old_level": old_level,
                "new_level": int(round(float(simple.GetMasterVolume()) * 100)),
                "muted": bool(simple.GetMute()),
                "candidates": candidates,
            }

        if action == "change":
            if delta is None:
                return {"error": {"type": "invalid_args", "message": "action=change requires delta"}}
            d = _clamp_int(int(delta), -100, 100)
            new_level = _clamp_int(old_level + d, 0, 100)
            simple.SetMasterVolume(new_level / 100.0, None)
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "delta": d,
                "old_level": old_level,
                "new_level": int(round(float(simple.GetMasterVolume()) * 100)),
                "muted": bool(simple.GetMute()),
                "candidates": candidates,
            }

        if action == "mute":
            simple.SetMute(1, None)
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "level": int(round(float(simple.GetMasterVolume()) * 100)),
                "muted": True,
                "candidates": candidates,
            }

        if action == "unmute":
            simple.SetMute(0, None)
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "level": int(round(float(simple.GetMasterVolume()) * 100)),
                "muted": False,
                "candidates": candidates,
            }

        if action == "toggle_mute":
            simple.SetMute(0 if old_muted else 1, None)
            return {
                "status": "ok",
                "scope": "app",
                "process": info.process_name,
                "display": info.display_name,
                "level": int(round(float(simple.GetMasterVolume()) * 100)),
                "muted": bool(simple.GetMute()),
                "candidates": candidates,
            }

        return {"error": {"type": "invalid_args", "message": f"Unknown action: {action}"}}
