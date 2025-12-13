"""
Window management tools for Windows desktop control.
"""
import os
import time
import ctypes
from typing import Dict, Any, List, Optional, Tuple
from wyzer.tools.tool_base import ToolBase

# Windows API constants
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9
WM_CLOSE = 0x0010

# Try to import pywin32, fallback to ctypes
try:
    import win32gui
    import win32con
    import win32process
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

# ctypes Windows API definitions
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


# Cache last-known window handles for better reliability with apps that have
# dynamic/empty titles (e.g., Spotify showing song titles).
_WINDOW_HANDLE_CACHE: Dict[str, Dict[str, Any]] = {}
_WINDOW_HANDLE_CACHE_TTL_S = 300.0


def _cache_key(kind: str, value: Optional[str]) -> Optional[str]:
    norm = (value or "").strip().lower()
    if not norm:
        return None
    return f"{kind}:{norm}"


def _cache_get(hwnd_key: Optional[str]) -> Optional[int]:
    if not hwnd_key:
        return None
    entry = _WINDOW_HANDLE_CACHE.get(hwnd_key)
    if not entry:
        return None
    try:
        if (time.time() - float(entry.get("ts", 0.0))) > _WINDOW_HANDLE_CACHE_TTL_S:
            _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
            return None
    except Exception:
        _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
        return None

    hwnd = entry.get("hwnd")
    if not isinstance(hwnd, int) or hwnd <= 0:
        _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
        return None

    try:
        if HAS_PYWIN32:
            if not win32gui.IsWindow(hwnd):
                _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
                return None
        else:
            if not user32.IsWindow(hwnd):
                _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
                return None
    except Exception:
        _WINDOW_HANDLE_CACHE.pop(hwnd_key, None)
        return None

    return hwnd


def _cache_put(hwnd: int, *, title: Optional[str], process: Optional[str], effective_process: Optional[str] = None) -> None:
    if not isinstance(hwnd, int) or hwnd <= 0:
        return
    ts = time.time()
    for key in (
        _cache_key("title", title),
        _cache_key("process", process),
        _cache_key("process", effective_process),
    ):
        if key:
            _WINDOW_HANDLE_CACHE[key] = {"hwnd": hwnd, "ts": ts}


def _infer_process_from_app_path(app_path: str) -> Optional[str]:
    if not app_path:
        return None
    base = os.path.basename(app_path).strip().lower()
    if base.endswith(".exe"):
        return base
    return None


def _try_resolve_phrase_to_process(phrase: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve a spoken app phrase (e.g., 'vs code') to a process name.

    Uses LocalLibrary resolve_target() to map phrase -> app exe path, then extracts
    the process filename (e.g., code.exe).
    """
    phrase_norm = (phrase or "").strip()
    if not phrase_norm:
        return None, None

    try:
        from wyzer.local_library import resolve_target
        resolved = resolve_target(phrase_norm)
    except Exception:
        return None, None

    if not isinstance(resolved, dict):
        return None, None

    if resolved.get("type") != "app":
        return None, resolved

    process_name = _infer_process_from_app_path(resolved.get("path", ""))
    return process_name, resolved


def _maybe_learn_alias(spoken_phrase: str, resolved: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Persist a high-confidence mapping spoken_phrase -> resolved target.

    Returns a small dict describing the write result, or None if skipped.
    """
    try:
        from wyzer.core.config import Config
        if not getattr(Config, "AUTO_ALIAS_ENABLED", False):
            return None
        min_conf = float(getattr(Config, "AUTO_ALIAS_MIN_CONFIDENCE", 0.85))
    except Exception:
        return None

    phrase_norm = (spoken_phrase or "").strip().lower()
    if not phrase_norm:
        return None

    if not isinstance(resolved, dict):
        return None

    target_type = resolved.get("type")
    target = resolved.get("path") or resolved.get("url")
    conf = float(resolved.get("confidence", 0.0) or 0.0)

    if target_type not in {"app", "folder", "file", "url"}:
        return None
    if not target:
        return None
    if conf < min_conf:
        return None

    try:
        from wyzer.local_library.indexer import get_cached_index
        index = get_cached_index()
        if phrase_norm in (index.get("aliases", {}) or {}):
            return None
        # Avoid writing aliases that are already canonical app keys.
        if target_type == "app" and phrase_norm in (index.get("apps", {}) or {}):
            return None
    except Exception:
        # If index lookup fails, still allow saving the alias.
        pass

    try:
        from wyzer.local_library.alias_manager import add_alias
        result = add_alias(phrase_norm, target_type, str(target), overwrite=False)
        return {
            "status": result.status,
            "alias": result.alias,
            "type": result.target_type,
            "target": result.target,
        }
    except Exception:
        return None


def _resolve_window_handle(
    *,
    title: Optional[str],
    process: Optional[str],
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[str]]:
    """Find a window, with LocalLibrary phrase fallback + auto-alias learning.

    Returns:
        (hwnd, learned_alias, effective_process)

    Notes:
        - effective_process may differ from the input "process" when a spoken phrase
          is resolved to an actual process name like "code.exe".
        - learned_alias is only non-None when a phrase was resolved with high
          confidence and persisted to aliases.json.
    """
    effective_process = process

    # First try cached handle(s) for stability across dynamic title changes.
    cached = _cache_get(_cache_key("process", process)) or _cache_get(_cache_key("title", title))
    if cached:
        return cached, None, effective_process

    hwnd = _find_window(title, process)
    if hwnd:
        _cache_put(hwnd, title=title, process=process, effective_process=effective_process)
        return hwnd, None, effective_process

    learned_alias = None

    # Try to interpret human phrases like "vs code" via LocalLibrary.
    if title and not process:
        inferred_process, resolved = _try_resolve_phrase_to_process(title)
        if inferred_process:
            effective_process = inferred_process
            hwnd = _find_window(None, inferred_process)
            if hwnd and isinstance(resolved, dict):
                learned_alias = _maybe_learn_alias(title, resolved)
                _cache_put(hwnd, title=title, process=process, effective_process=effective_process)
        elif isinstance(resolved, dict) and resolved.get("matched_name"):
            # If the app resolved to a Start Menu shortcut (.lnk), we may not know
            # the process. Use the matched/canonical app name as a title hint.
            hwnd = _find_window(str(resolved.get("matched_name")), None)
            if hwnd:
                learned_alias = _maybe_learn_alias(title, resolved)
                _cache_put(hwnd, title=title, process=process, effective_process=effective_process)

    elif process and not title:
        inferred_process, resolved = _try_resolve_phrase_to_process(process)
        if inferred_process:
            effective_process = inferred_process
            hwnd = _find_window(None, inferred_process)
            if hwnd and isinstance(resolved, dict):
                learned_alias = _maybe_learn_alias(process, resolved)
                _cache_put(hwnd, title=title, process=process, effective_process=effective_process)
        elif isinstance(resolved, dict) and resolved.get("matched_name"):
            hwnd = _find_window(str(resolved.get("matched_name")), None)
            if hwnd:
                learned_alias = _maybe_learn_alias(process, resolved)
                _cache_put(hwnd, title=title, process=process, effective_process=effective_process)

    if hwnd:
        _cache_put(hwnd, title=title, process=process, effective_process=effective_process)
    return hwnd, learned_alias, effective_process


def _enumerate_windows() -> List[Dict[str, Any]]:
    """
    Enumerate all visible windows.
    
    Returns:
        List of window info dicts
    """
    windows = []
    
    if HAS_PYWIN32:
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd) or ""
                pid = 0
                process_name = ""
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    process_name = _get_process_name_pywin32(pid)
                except:
                    pid = 0
                    process_name = ""

                # Some apps (notably UWP/Chromium-hosted windows) can have an empty
                # title even when clearly visible. Include these so process matching works.
                if title or process_name:
                    windows.append({
                        "hwnd": hwnd,
                        "title": title,
                        "pid": pid,
                        "process": process_name
                    })
            return True
        
        win32gui.EnumWindows(callback, None)
    else:
        # ctypes fallback
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        
        def callback(hwnd, lParam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                title = ""
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value or ""

                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                process_name = _get_process_name_ctypes(pid.value)

                if title or process_name:
                    windows.append({
                        "hwnd": hwnd,
                        "title": title,
                        "pid": pid.value,
                        "process": process_name
                    })
            return True
        
        enum_func = EnumWindowsProc(callback)
        user32.EnumWindows(enum_func, 0)
    
    return windows


def _get_process_name_pywin32(pid: int) -> str:
    """Get process name from PID using pywin32"""
    try:
        import win32api
        import win32process
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
        try:
            exe = win32process.GetModuleFileNameEx(handle, 0)
            return exe.split("\\")[-1].lower()
        finally:
            win32api.CloseHandle(handle)
    except:
        return ""


def _get_process_name_ctypes(pid: int) -> str:
    """Get process name from PID using ctypes"""
    try:
        # Prefer QUERY_LIMITED_INFORMATION, which works for more processes on modern Windows.
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        
        if handle:
            try:
                buff = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                
                # QueryFullProcessImageNameW
                if kernel32.QueryFullProcessImageNameW(handle, 0, buff, ctypes.byref(size)):
                    return buff.value.split("\\")[-1].lower()
            finally:
                kernel32.CloseHandle(handle)
    except:
        pass
    
    return ""


def _find_window(title: Optional[str] = None, process: Optional[str] = None) -> Optional[int]:
    """
    Find window by title or process name.
    
    Args:
        title: Window title (substring match, case-insensitive)
        process: Process name (substring match, case-insensitive)
        
    Returns:
        Window handle (hwnd) or None if not found
    """
    windows = _enumerate_windows()

    title_norm = (title or "").strip().lower()
    process_norm = (process or "").strip().lower()

    # Many apps (e.g., Spotify) show dynamic titles (song names) that don't include
    # the app name. If the caller provided only a title, treat it as a process hint too.
    process_hints: List[str] = []
    if title_norm and not process_norm:
        base = title_norm
        if base.endswith(".exe"):
            base = base[:-4]
        compact = base.replace(" ", "")
        process_hints = [
            base,
            compact,
            f"{base}.exe",
            f"{compact}.exe",
        ]
        # Deduplicate while preserving order.
        seen = set()
        process_hints = [h for h in process_hints if h and not (h in seen or seen.add(h))]

    best_hwnd: Optional[int] = None
    best_score = -1

    for window in windows:
        window_title = (window.get("title") or "").lower()
        window_process = (window.get("process") or "").lower()

        score = -1

        if title_norm and title_norm in window_title:
            # Strongest signal: explicit title match.
            score = 100

        if process_norm and process_norm in window_process:
            # Strong signal: explicit process match.
            score = max(score, 90)

        if process_hints and any(h in window_process for h in process_hints):
            # Heuristic fallback: treat title phrase as process hint.
            score = max(score, 80)

        if score > best_score:
            best_score = score
            best_hwnd = window.get("hwnd")

    return best_hwnd if best_score >= 0 else None


def _get_window_info(hwnd: int) -> Dict[str, Any]:
    """Get window information"""
    windows = _enumerate_windows()
    for window in windows:
        if window["hwnd"] == hwnd:
            return window
    return {}


class FocusWindowTool(ToolBase):
    """Tool to focus/activate a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "focus_window"
        self._description = "Focus/activate a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd, learned_alias, _ = _resolve_window_handle(title=title, process=process)
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Focus window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
            else:
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetForegroundWindow(hwnd)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "focused",
                "matched": window_info,
                "learned_alias": learned_alias,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MinimizeWindowTool(ToolBase):
    """Tool to minimize a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "minimize_window"
        self._description = "Minimize a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd, learned_alias, _ = _resolve_window_handle(title=title, process=process)
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Minimize window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            else:
                user32.ShowWindow(hwnd, SW_MINIMIZE)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "minimized",
                "matched": window_info,
                "learned_alias": learned_alias,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MaximizeWindowTool(ToolBase):
    """Tool to maximize a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "maximize_window"
        self._description = "Maximize a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd, learned_alias, _ = _resolve_window_handle(title=title, process=process)
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Maximize window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            else:
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "maximized",
                "matched": window_info,
                "learned_alias": learned_alias,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class CloseWindowTool(ToolBase):
    """Tool to close a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "close_window"
        self._description = "Close a window by title or process name (requires allowlist for force close)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force close (requires config flag and allowlist)",
                    "default": False
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.core.config import Config
        
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        force = kwargs.get("force", False)
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        # Check force close permission
        if force and not getattr(Config, "ENABLE_FORCE_CLOSE", False):
            return {
                "error": {
                    "type": "permission_denied",
                    "message": "Force close is disabled in configuration"
                },
                "latency_ms": 0
            }
        
        try:
            hwnd, learned_alias, effective_process = _resolve_window_handle(title=title, process=process)
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            window_info = _get_window_info(hwnd)
            
            # Check allowlist for close
            if effective_process:
                allowed = getattr(Config, "ALLOWED_PROCESSES_TO_CLOSE", [])
                if allowed and effective_process.lower() not in [p.lower() for p in allowed]:
                    end_time = time.perf_counter()
                    return {
                        "error": {
                            "type": "permission_denied",
                            "message": f"Process '{effective_process}' not in allowed list"
                        },
                        "latency_ms": int((end_time - start_time) * 1000)
                    }
            
            # Close window
            if HAS_PYWIN32:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            else:
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            
            end_time = time.perf_counter()
            
            return {
                "status": "closed",
                "matched": window_info,
                "learned_alias": learned_alias,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MoveWindowToMonitorTool(ToolBase):
    """Tool to move window to specific monitor"""
    
    def __init__(self):
        super().__init__()
        self._name = "move_window_to_monitor"
        self._description = "Move a window to a specific monitor with positioning (use 'primary' or monitor number)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                },
                "monitor": {
                    "description": "Monitor number (1-based) or 'primary' for primary monitor",
                    "oneOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "string", "enum": ["primary"]}
                    ]
                },
                "position": {
                    "type": "string",
                    "enum": ["left", "right", "center", "maximize"],
                    "description": "Position on monitor",
                    "default": "maximize"
                }
            },
            "required": ["monitor"],
            "additionalProperties": False
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()

        title = kwargs.get("title")
        process = kwargs.get("process")
        monitor_spec = kwargs.get("monitor", 0)
        position = kwargs.get("position", "maximize")

        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }

        try:
            monitors = _enumerate_monitors()

            # Handle "primary" keyword
            if isinstance(monitor_spec, str) and monitor_spec.lower() == "primary":
                primary_monitor = None
                for mon in monitors:
                    if mon.get("primary", False):
                        primary_monitor = mon
                        break

                if primary_monitor is None:
                    return {
                        "error": {
                            "type": "invalid_monitor",
                            "message": "No primary monitor found"
                        },
                        "latency_ms": int((time.perf_counter() - start_time) * 1000)
                    }

                monitor = primary_monitor
            else:
                # Natural speech: treat monitor numbers as 1-based ("monitor 1" => first monitor).
                monitor_index = int(monitor_spec)
                if monitor_index >= 1:
                    monitor_index -= 1

                if monitor_index >= len(monitors):
                    return {
                        "error": {
                            "type": "invalid_monitor",
                            "message": f"Monitor {monitor_spec} out of range (1-{len(monitors)})"
                        },
                        "latency_ms": int((time.perf_counter() - start_time) * 1000)
                    }

                monitor = monitors[monitor_index]

            hwnd, learned_alias, _ = _resolve_window_handle(title=title, process=process)
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }

            mon_x = monitor["x"]
            mon_y = monitor["y"]
            mon_w = monitor["width"]
            mon_h = monitor["height"]

            if position == "maximize":
                if HAS_PYWIN32:
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                else:
                    user32.ShowWindow(hwnd, SW_RESTORE)

                user32.SetWindowPos(hwnd, 0, mon_x, mon_y, mon_w, mon_h, 0)

                if HAS_PYWIN32:
                    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                else:
                    user32.ShowWindow(hwnd, SW_MAXIMIZE)

            elif position == "left":
                win_w = mon_w // 2
                win_h = mon_h
                user32.SetWindowPos(hwnd, 0, mon_x, mon_y, win_w, win_h, 0)

            elif position == "right":
                win_w = mon_w // 2
                win_h = mon_h
                win_x = mon_x + win_w
                user32.SetWindowPos(hwnd, 0, win_x, mon_y, win_w, win_h, 0)

            elif position == "center":
                win_w = int(mon_w * 0.7)
                win_h = int(mon_h * 0.7)
                win_x = mon_x + (mon_w - win_w) // 2
                win_y = mon_y + (mon_h - win_h) // 2
                user32.SetWindowPos(hwnd, 0, win_x, win_y, win_w, win_h, 0)

            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()

            return {
                "status": "moved",
                "matched": window_info,
                "monitor": monitor,
                "position": position,
                "learned_alias": learned_alias,
                "latency_ms": int((end_time - start_time) * 1000)
            }

        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


def _enumerate_monitors() -> List[Dict[str, Any]]:
    """Enumerate all monitors"""
    monitors = []
    
    if HAS_PYWIN32:
        import win32api
        
        for i, monitor in enumerate(win32api.EnumDisplayMonitors()):
            info = win32api.GetMonitorInfo(monitor[0])
            mon_rect = info["Monitor"]
            
            monitors.append({
                "index": i,
                "x": mon_rect[0],
                "y": mon_rect[1],
                "width": mon_rect[2] - mon_rect[0],
                "height": mon_rect[3] - mon_rect[1],
                "primary": info.get("Flags", 0) == 1
            })
    else:
        # ctypes fallback
        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long * 4), ctypes.c_double)
        
        def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            rect = lprcMonitor.contents
            monitors.append({
                "index": len(monitors),
                "x": rect[0],
                "y": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
                "primary": rect[0] == 0 and rect[1] == 0  # Heuristic
            })
            return 1
        
        enum_func = MonitorEnumProc(callback)
        user32.EnumDisplayMonitors(0, 0, enum_func, 0)
    
    return monitors
