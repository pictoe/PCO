"""Audio output device switching for Windows.

Adds a tool that switches the system default playback (render) device using
fuzzy matching (so imperfect spoken names still work).

Implementation notes:
- Device listing uses pycaw.
- Default-device switching uses the (undocumented) PolicyConfig COM interface.
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from wyzer.tools.tool_base import ToolBase


def _normalize_for_tokens(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _singularize_token(token: str) -> str:
    tok = (token or "").strip().lower()
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _token_match_ratio(query: str, candidate: str) -> float:
    q = _normalize_for_tokens(query)
    c = _normalize_for_tokens(candidate)
    if not q or not c:
        return 0.0

    q_tokens = [_singularize_token(t) for t in q.split() if t.strip()]
    c_tokens = [_singularize_token(t) for t in c.split() if t.strip()]
    if not q_tokens or not c_tokens:
        return 0.0

    def matches(qt: str) -> bool:
        for ct in c_tokens:
            if qt == ct:
                return True
            # allow prefix/substr matches for brand/device terms
            if len(qt) >= 3 and (qt in ct or ct in qt):
                return True
        return False

    hit = sum(1 for qt in q_tokens if matches(qt))
    return hit / max(1, len(q_tokens))


def _fuzzy_score(query: str, candidate: str) -> int:
    """Return a 0-100 similarity score."""
    q = _normalize_for_tokens(query)
    c = _normalize_for_tokens(candidate)

    if not q or not c:
        return 0
    if q == c:
        return 100

    ratio = SequenceMatcher(None, q, c).ratio()

    q_tokens = {_singularize_token(t) for t in q.split() if t.strip()}
    c_tokens = {_singularize_token(t) for t in c.split() if t.strip()}
    union = q_tokens | c_tokens
    inter = q_tokens & c_tokens
    jaccard = (len(inter) / len(union)) if union else 0.0

    qc = _normalize_compact(query)
    cc = _normalize_compact(candidate)
    partial = 0.0
    if qc and cc:
        if qc in cc:
            partial = min(1.0, len(qc) / max(1, len(cc)))
        else:
            match = SequenceMatcher(None, qc, cc).find_longest_match()
            if match.size > 0:
                partial = min(1.0, match.size / max(1, len(qc)))

    token_hit = _token_match_ratio(query, candidate)

    score = max(ratio, jaccard, partial, token_hit)

    # Strong boost when the candidate contains all query tokens.
    if token_hit >= 0.999:
        score = max(score, 0.90)
    elif token_hit >= 0.66:
        score = max(score, 0.78)

    return int(round(score * 100))


def _safe_device_name(device: Any) -> str:
    for attr in ("FriendlyName", "friendly_name", "name"):
        try:
            val = getattr(device, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            continue
    return "(unknown device)"


def _safe_device_id(device: Any) -> Optional[str]:
    for attr in ("id", "Id", "device_id"):
        try:
            val = getattr(device, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            continue
    return None


def _device_id_is_render(device_id: Optional[str]) -> Optional[bool]:
    """Detect render vs capture by Windows endpoint id prefix.

    Typical endpoint IDs:
      - Render/output: {0.0.0.00000000}.{GUID}
      - Capture/input: {0.0.1.00000000}.{GUID}

    Returns True/False, or None if unknown.
    """
    if not device_id:
        return None
    s = device_id.strip().lower()
    if s.startswith("{0.0.0."):
        return True
    if s.startswith("{0.0.1."):
        return False
    return None


def _is_render_device(device: Any) -> bool:
    dev_id = _safe_device_id(device)
    by_id = _device_id_is_render(dev_id)
    if by_id is not None:
        return bool(by_id)

    # Fallback: some pycaw versions expose EDataFlow.
    try:
        from pycaw.pycaw import EDataFlow

        flow = getattr(device, "data_flow", None)
        if flow is None:
            flow = getattr(device, "DataFlow", None)
        if flow is None:
            return True
        return flow == EDataFlow.eRender
    except Exception:
        return True


def _list_render_devices() -> List[Any]:
    import warnings
    from pycaw.pycaw import AudioUtilities

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"COMError attempting to get property.*",
            category=UserWarning,
        )
        devices = AudioUtilities.GetAllDevices()

    render: List[Any] = []
    seen_ids: set[str] = set()

    for d in devices:
        if not _is_render_device(d):
            continue
        name = _safe_device_name(d)
        if not name or name.strip().lower() in {"none", "(unknown device)"}:
            continue
        dev_id = _safe_device_id(d) or ""
        if dev_id and dev_id in seen_ids:
            continue
        if dev_id:
            seen_ids.add(dev_id)
        render.append(d)

    return render


def _pick_best_device(query: str, devices: List[Any]) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []

    for dev in devices:
        name = _safe_device_name(dev)
        score = _fuzzy_score(query, name)
        scored.append(
            {
                "device": dev,
                "name": name,
                "id": _safe_device_id(dev),
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    top_for_debug = [
        {
            "name": s["name"],
            "score": int(s["score"]),
            "id": s["id"],
        }
        for s in scored[:8]
    ]

    if not scored:
        return None, top_for_debug

    return scored[0]["device"], top_for_debug


def _set_default_audio_endpoint(device_id: str) -> None:
    """Set default render device for console/multimedia/communications."""
    import ctypes
    from ctypes import POINTER

    import comtypes
    from comtypes import GUID, COMMETHOD, HRESULT
    from comtypes.client import CreateObject

    CLSID_PolicyConfigClient = GUID("{870AF99C-171D-4F9E-AF0D-E63DF40C2BC9}")

    class IPolicyConfig(comtypes.IUnknown):
        _iid_ = GUID("{F8679F50-850A-41CF-9C72-430F290290C8}")

    class IPolicyConfigVista(comtypes.IUnknown):
        _iid_ = GUID("{568B9108-44BF-40B4-9006-86AFE5B5A620}")

    # Keep placeholder arg types conservative; vtable order matters.
    methods = [
        COMMETHOD([], HRESULT, "GetMixFormat", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["out"], POINTER(ctypes.c_void_p), "ppFormat")),
        COMMETHOD([], HRESULT, "GetDeviceFormat", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_int, "bDefault"), (["out"], POINTER(ctypes.c_void_p), "ppFormat")),
        COMMETHOD([], HRESULT, "ResetDeviceFormat", (["in"], ctypes.c_wchar_p, "pwstrDeviceId")),
        COMMETHOD([], HRESULT, "SetDeviceFormat", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_void_p, "pEndpointFormat"), (["in"], ctypes.c_void_p, "pMixFormat")),
        COMMETHOD([], HRESULT, "GetProcessingPeriod", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_int, "bDefault"), (["out"], POINTER(ctypes.c_longlong), "pDefaultPeriod"), (["out"], POINTER(ctypes.c_longlong), "pMinimumPeriod")),
        COMMETHOD([], HRESULT, "SetProcessingPeriod", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], POINTER(ctypes.c_longlong), "pDefaultPeriod"), (["in"], POINTER(ctypes.c_longlong), "pMinimumPeriod")),
        COMMETHOD([], HRESULT, "GetShareMode", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["out"], POINTER(ctypes.c_void_p), "pMode")),
        COMMETHOD([], HRESULT, "SetShareMode", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_void_p, "pMode")),
        COMMETHOD([], HRESULT, "GetPropertyValue", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_void_p, "pKey"), (["out"], POINTER(ctypes.c_void_p), "pv")),
        COMMETHOD([], HRESULT, "SetPropertyValue", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_void_p, "pKey"), (["in"], ctypes.c_void_p, "pv")),
        COMMETHOD([], HRESULT, "SetDefaultEndpoint", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_int, "role")),
        COMMETHOD([], HRESULT, "SetEndpointVisibility", (["in"], ctypes.c_wchar_p, "pwstrDeviceId"), (["in"], ctypes.c_int, "bVisible")),
    ]

    IPolicyConfig._methods_ = methods
    IPolicyConfigVista._methods_ = methods

    def try_set(interface_cls) -> bool:
        try:
            policy = CreateObject(CLSID_PolicyConfigClient, interface=interface_cls)
            for role in (0, 1, 2):
                hr = policy.SetDefaultEndpoint(device_id, int(role))
                if hr not in (0, None):
                    return False
            return True
        except Exception:
            return False

    comtypes.CoInitialize()
    try:
        if try_set(IPolicyConfig):
            return
        if try_set(IPolicyConfigVista):
            return
        raise OSError("Failed to set default endpoint via PolicyConfig")
    finally:
        comtypes.CoUninitialize()


class SetAudioOutputDeviceTool(ToolBase):
    """Switch the system default audio output device."""

    def __init__(self):
        super().__init__()
        self._name = "set_audio_output_device"
        self._description = (
            "Switch the default Windows audio output device by fuzzy name match "
            "(e.g., 'vizo speaker', 'logitech headset')."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Desired output device name (fuzzy match allowed)",
                },
                "min_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 55,
                    "description": "Minimum fuzzy match score (0-100) to accept",
                },
            },
            "required": ["device"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()

        query = (kwargs.get("device") or "").strip()
        min_score = int(kwargs.get("min_score", 55))

        if not query:
            return {
                "error": {"type": "invalid_device_query", "message": "Device name cannot be empty"}
            }

        try:
            devices = _list_render_devices()
            if not devices:
                end_time = time.perf_counter()
                return {
                    "error": {"type": "no_devices", "message": "No audio output devices were found"},
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            best, top = _pick_best_device(query, devices)
            if best is None:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "not_found",
                        "message": f"Could not find an audio output device matching '{query}'",
                    },
                    "candidates": top,
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            best_name = _safe_device_name(best)
            best_id = _safe_device_id(best)
            score = _fuzzy_score(query, best_name)

            if score < min_score:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "low_confidence",
                        "message": f"Best match '{best_name}' scored {score}/100 which is below the threshold ({min_score}/100).",
                    },
                    "chosen": {"name": best_name, "id": best_id, "score": score},
                    "candidates": top,
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            if not best_id:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "device_id_missing",
                        "message": f"Matched '{best_name}' but could not read its device id.",
                    },
                    "candidates": top,
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            _set_default_audio_endpoint(best_id)

            end_time = time.perf_counter()
            return {
                "status": "ok",
                "action": "set_audio_output_device",
                "requested": query,
                "chosen": {"name": best_name, "id": best_id, "score": score},
                "candidates": top,
                "latency_ms": int((end_time - start_time) * 1000),
            }

        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {"type": "execution_error", "message": str(e)},
                "latency_ms": int((end_time - start_time) * 1000),
            }
