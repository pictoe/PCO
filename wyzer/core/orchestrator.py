"""
Orchestrator for Wyzer AI Assistant - Phase 6
Coordinates LLM reasoning and tool execution.
Supports multi-intent commands (Phase 6 enhancement).
"""
import json
import time
import urllib.request
import urllib.error
import re
import socket
import shlex
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args
from wyzer.local_library import resolve_target
from wyzer.core.intent_plan import (
    normalize_plan,
    Intent,
    validate_intents,
    ExecutionResult,
    ExecutionSummary
)


_FASTPATH_SPLIT_RE = re.compile(r"\b(?:and then|then|and)\b", re.IGNORECASE)
_FASTPATH_COMMA_SPLIT_RE = re.compile(r"\s*,\s*")
_FASTPATH_SEMI_SPLIT_RE = re.compile(r"\s*;\s*")
_FASTPATH_COMMAND_TOKEN_RE = re.compile(
    r"\b(?:tool|run|execute|open|launch|start|close|exit|quit|focus|activate|switch\s+to|minimize|maximize|fullscreen|move|pause|play|resume|mute|unmute|volume\s+up|volume\s+down|turn\s+up|turn\s+down|louder|quieter|set\s+audio|switch\s+audio|change\s+audio|refresh\s+library|rebuild\s+library|weather|forecast|location|system\s+info|system\s+information|monitor\s+info|what\s+time\s+is\s+it|my\s+location|where\s+am\s+i|next\s+(?:track|song)|previous\s+(?:track|song)|prev\s+track)\b",
    re.IGNORECASE,
)

# Small allowlist of queries that are overwhelmingly likely to mean a website.
# Keep conservative to preserve the "only if unambiguous" rule.
_FASTPATH_COMMON_WEBSITES = {
    "youtube",
    "github",
    "google",
    "gmail",
    "wikipedia",
    "reddit",
}

_FASTPATH_EXPLICIT_TOOL_RE = re.compile(r"^(?:tool|run|execute)\s+(?P<tool>[a-zA-Z0-9_]+)(?:\s+(?P<rest>.*))?$", re.IGNORECASE)

# Module-level singleton registry
_registry = None
_logger = None


def get_logger_instance():
    """Get or create logger instance"""
    global _logger
    if _logger is None:
        _logger = get_logger()
    return _logger


def get_registry():
    """Get or create the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def handle_user_text(text: str) -> Dict[str, Any]:
    """
    Handle user text input with optional multi-intent tool execution.
    
    Args:
        text: User's input text
        
    Returns:
        Dict with "reply", "latency_ms", and optional "execution_summary" keys
    """
    start_time = time.perf_counter()
    logger = get_logger_instance()
    
    try:
        registry = get_registry()

        # Fast-path: bypass LLM for high-confidence tool commands.
        fast_intents = _try_fastpath_intents(text, registry)
        if fast_intents:
            tool_names = [intent.tool for intent in fast_intents]
            logger.info(f"[FASTPATH] Executing {len(fast_intents)} intent(s): {', '.join(tool_names)}")

            try:
                validate_intents(fast_intents, registry)
            except ValueError as e:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": f"I cannot execute that request: {str(e)}",
                    "latency_ms": latency_ms,
                }

            execution_summary = _execute_intents(fast_intents, registry)
            reply = _format_fastpath_reply(text, fast_intents, execution_summary)

            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            return {
                "reply": reply,
                "latency_ms": latency_ms,
                "execution_summary": {
                    "ran": [
                        {
                            "tool": r.tool,
                            "ok": r.ok,
                            "result": r.result,
                            "error": r.error,
                        }
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early,
                },
            }
        
        # First LLM call: interpret user intent(s)
        llm_response = _call_llm(text, registry)
        
        # Normalize LLM response to standard IntentPlan format
        intent_plan = normalize_plan(llm_response)

        # Heuristic rewrite: fix common LLM confusion where a game/app name
        # gets turned into an open_website URL (e.g., "Rocket League" -> rocketleague.com)
        _rewrite_open_website_intents(text, intent_plan.intents)
        
        # Check if there are any intents to execute
        if intent_plan.intents:
            # Log parsed plan (tool names only)
            tool_names = [intent.tool for intent in intent_plan.intents]
            logger.info(f"[INTENT PLAN] Executing {len(intent_plan.intents)} intent(s): {', '.join(tool_names)}")
            
            # Validate all intents before execution
            try:
                validate_intents(intent_plan.intents, registry)
            except ValueError as e:
                # Validation failed - return error
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": f"I cannot execute that request: {str(e)}",
                    "latency_ms": latency_ms
                }
            
            # Execute intents sequentially
            execution_summary = _execute_intents(intent_plan.intents, registry)
            
            # Second LLM call: generate final reply with execution results
            final_response = _call_llm_with_execution_summary(
                text, execution_summary, registry
            )
            
            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "reply": final_response.get("reply", "I executed the action."),
                "latency_ms": latency_ms,
                "execution_summary": {
                    "ran": [
                        {
                            "tool": r.tool,
                            "ok": r.ok,
                            "result": r.result,
                            "error": r.error
                        }
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early
                }
            }
        else:
            # No intents needed, return direct reply
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "reply": intent_plan.reply or llm_response.get("reply", ""),
                "latency_ms": latency_ms
            }
            
    except Exception as e:
        # Safe fallback on any error
        logger.error(f"[ORCHESTRATOR ERROR] {str(e)}")
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "reply": f"I encountered an error: {str(e)}",
            "latency_ms": latency_ms
        }


def _execute_intents(intents, registry) -> ExecutionSummary:
    """
    Execute multiple intents sequentially and collect results.
    
    Args:
        intents: List of Intent objects to execute
        registry: Tool registry
        
    Returns:
        ExecutionSummary with results of all executed intents
    """
    logger = get_logger_instance()
    results = []
    stopped_early = False
    
    for idx, intent in enumerate(intents):
        logger.info(f"[INTENT {idx + 1}/{len(intents)}] Executing: {intent.tool}")
        
        # Execute the tool
        tool_result = _execute_tool(registry, intent.tool, intent.args)
        
        # Check if execution was successful
        has_error = "error" in tool_result

        error_type = None
        if has_error:
            try:
                error_type = (tool_result.get("error") or {}).get("type")
            except Exception:
                error_type = None
        
        # Create execution result
        exec_result = ExecutionResult(
            tool=intent.tool,
            ok=not has_error,
            result=tool_result if not has_error else None,
            error=str(tool_result.get("error")) if has_error else None
        )
        
        results.append(exec_result)
        
        # If error occurred and continue_on_error is False, stop execution
        if has_error and not intent.continue_on_error:
            # Focus is often an optional first step before window operations.
            # If it fails to locate the window, still try subsequent actions.
            if intent.tool == "focus_window" and idx < (len(intents) - 1) and error_type == "window_not_found":
                logger.info(f"[INTENT {idx + 1}/{len(intents)}] Focus failed (window_not_found), continuing")
                continue
            logger.info(f"[INTENT {idx + 1}/{len(intents)}] Failed, stopping execution")
            stopped_early = True
            break
        
        logger.info(f"[INTENT {idx + 1}/{len(intents)}] {'Success' if not has_error else 'Failed (continuing)'}")
    
    return ExecutionSummary(ran=results, stopped_early=stopped_early)


def _normalize_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"")


def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,2})\b", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_scalar_value(raw: str) -> Any:
    v = (raw or "").strip()
    if not v:
        return ""

    vl = v.lower()
    if vl in {"true", "yes", "y", "on"}:
        return True
    if vl in {"false", "no", "n", "off"}:
        return False

    # int
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except Exception:
            pass

    # float
    if re.fullmatch(r"-?\d+\.\d+", v):
        try:
            return float(v)
        except Exception:
            pass

    return v


def _pick_single_arg_key_from_schema(schema: Dict[str, Any]) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None

    required = schema.get("required")
    if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
        return required[0]

    if len(props) == 1:
        return next(iter(props.keys()))

    return None


def _try_parse_explicit_tool_clause(clause: str, registry) -> Optional[Intent]:
    """Parse an explicit tool invocation.

    Supported forms (case-insensitive):
      - tool <tool_name> {"json": "args"}
      - tool <tool_name> key=value key2="value with spaces"
      - tool <tool_name> <free text>    (only when schema has a single arg)
      - optional: continue_on_error=true
    """
    text = (clause or "").strip()
    if not text:
        return None

    m = _FASTPATH_EXPLICIT_TOOL_RE.match(text)
    if not m:
        return None

    tool_name = (m.group("tool") or "").strip()
    if not tool_name:
        return None

    # Tool names in registry are lowercase snake_case.
    tool_name = tool_name.strip()
    if not registry.has_tool(tool_name):
        return None

    rest = (m.group("rest") or "").strip()
    args: Dict[str, Any] = {}
    continue_on_error = False

    if rest:
        # JSON args dict (optionally includes continue_on_error)
        if rest.startswith("{") and rest.endswith("}"):
            try:
                payload = json.loads(rest)
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None

            # Allow either direct args or nested args.
            if "args" in payload and isinstance(payload.get("args"), dict):
                args = dict(payload.get("args") or {})
            else:
                args = dict(payload)

            if isinstance(args.get("continue_on_error"), bool):
                continue_on_error = bool(args.pop("continue_on_error"))
            return Intent(tool=tool_name, args=args, continue_on_error=continue_on_error)

        # key=value tokens / quoted strings
        try:
            tokens = shlex.split(rest, posix=False)
        except Exception:
            tokens = rest.split()

        tail: List[str] = []
        for tok in tokens:
            if "=" not in tok:
                tail.append(tok)
                continue
            k, v = tok.split("=", 1)
            k = (k or "").strip()
            v = (v or "").strip().strip('"').strip("'")
            if not k:
                return None

            if k in {"continue_on_error", "continue", "co"}:
                continue_on_error = bool(_parse_scalar_value(v))
                continue

            args[k] = _parse_scalar_value(v)

        if tail:
            # If we already saw key=value args, trailing free-text is ambiguous.
            if args:
                return None
            tool = registry.get(tool_name)
            schema = getattr(tool, "args_schema", {}) if tool is not None else {}
            key = _pick_single_arg_key_from_schema(schema)
            if not key:
                return None
            args[key] = _strip_trailing_punct(" ".join(tail)).strip()
            if not args[key]:
                return None

    return Intent(tool=tool_name, args=args, continue_on_error=continue_on_error)


def _extract_days(text: str) -> Optional[int]:
    """Extract a day count from phrases like 'next 5 days' or '5 day forecast'."""
    if not text:
        return None
    m = re.search(r"\b(?:next\s+)?(\d{1,2})\s+day(?:s)?\b", text.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_location_after_in(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(?:in|for)\s+(.+)$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    loc = _strip_trailing_punct(m.group(1)).strip()
    return loc or None


def _parse_audio_device_target(text: str) -> Optional[str]:
    """Extract device name from explicit audio output switching commands."""
    if not text:
        return None

    t = text.strip()
    tl = t.lower()

    # Require explicit audio/output context to avoid accidental triggers.
    explicit_markers = [
        "audio output",
        "output device",
        "audio device",
        "playback device",
        "default audio",
        "default output",
        "speakers",
        "headphones",
        "headset",
        "earbuds",
    ]
    if not any(m in tl for m in explicit_markers):
        return None

    # Common patterns with an explicit target.
    patterns = [
        r"^(?:set|switch|change)\s+(?:the\s+)?(?:default\s+)?(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)(?:\s+device)?\s+(?:to|as)\s+(.+)$",
        r"^(?:set|switch|change)\s+(?:to)\s+(.+?)\s+(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)$",
        r"^(?:use)\s+(.+?)\s+(?:as)\s+(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)$",
        r"^(?:set|switch|change)\s+(?:speakers|headphones|headset|earbuds)\s+(?:to)\s+(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, tl)
        if not m:
            continue
        dev = _strip_trailing_punct(m.group(1)).strip()
        if not dev:
            return None
        # Reject placeholder / ambiguous targets.
        if dev in {"audio", "output", "device", "speakers", "headphones", "headset", "earbuds"}:
            return None
        return dev

    # A very explicit fallback: "set audio output to <dev>" somewhere in the clause.
    m = re.search(r"\b(?:audio\s+output|output\s+device|playback\s+device)\s+(?:to|as)\s+(.+)$", tl)
    if m:
        dev = _strip_trailing_punct(m.group(1)).strip()
        if dev and dev not in {"audio", "output", "device"}:
            return dev

    return None


def _looks_like_url(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t:
        return False
    if t.startswith("http://") or t.startswith("https://") or t.startswith("www."):
        return True
    # Very small heuristic for domain-ish strings.
    return bool(re.search(r"\.[a-z]{2,}$", t))


def _try_fastpath_intents(user_text: str, registry) -> Optional[List[Intent]]:
    """Return a list of intents when the command is unambiguous.

    This is a deterministic, conservative parser intended to bypass the LLM.
    If unsure, return None and let the LLM handle it.
    """
    raw = (user_text or "").strip()
    if not raw:
        return None

    # Split basic multi-step commands: "pause music and mute volume".
    parts = [p.strip() for p in _FASTPATH_SPLIT_RE.split(raw) if p and p.strip()]

    # Expand comma-separated command lists inside each part, so mixed separators work:
    #   "open spotify, then open chrome and open youtube"
    # We keep this conservative to avoid breaking things like "Paris, France".
    expanded: List[str] = []
    for p in parts:
        # Always allow ';' as an explicit command separator.
        if ";" in p:
            expanded.extend([s.strip() for s in _FASTPATH_SEMI_SPLIT_RE.split(p) if s and s.strip()])
            continue

        if "," not in p:
            expanded.append(p)
            continue

        tl = p.lower()
        # Avoid splitting common location strings in weather/forecast commands.
        comma_split_allowed = not (("weather" in tl or "forecast" in tl) and " in " in tl)
        token_hits = len(_FASTPATH_COMMAND_TOKEN_RE.findall(p))

        if comma_split_allowed and token_hits >= 2:
            expanded.extend([s.strip() for s in _FASTPATH_COMMA_SPLIT_RE.split(p) if s and s.strip()])
        else:
            expanded.append(p)

    parts = expanded

    if not parts:
        return None
    if len(parts) > 5:
        return None

    intents: List[Intent] = []
    for part in parts:
        part_intents = _fastpath_parse_clause(part)
        if not part_intents:
            return None
        intents.extend(part_intents)
        if len(intents) > 5:
            return None

    # Final conservative sanity: ensure every referenced tool exists.
    if not intents:
        return None
    if any(not registry.has_tool(i.tool) for i in intents):
        return None
    return intents


def _fastpath_parse_clause(clause: str) -> Optional[List[Intent]]:
    """Parse a single command clause into intents."""
    c_raw = _strip_trailing_punct(clause)
    c = (c_raw or "").strip()
    if not c:
        return None

    c_lower = c.lower()

    # Explicit tool invocation supports ANY tool in the registry.
    explicit = _try_parse_explicit_tool_clause(c, registry=get_registry())
    if explicit:
        return [explicit]

    # --- Info tools ---
    if re.fullmatch(r"(what\s+time\s+is\s+it|what\s+is\s+the\s+time|time)\??", c_lower):
        return [Intent(tool="get_time", args={})]

    if "system info" in c_lower or "system information" in c_lower or re.fullmatch(r"system\s+info", c_lower):
        return [Intent(tool="get_system_info", args={})]

    if "monitor info" in c_lower or "monitors" == c_lower or re.fullmatch(r"monitor\s+info", c_lower):
        return [Intent(tool="monitor_info", args={})]

    if re.fullmatch(r"(where\s+am\s+i|what\s+is\s+my\s+location|my\s+location)\??", c_lower):
        return [Intent(tool="get_location", args={})]

    # Weather / forecast (safe, internet required)
    if "weather" in c_lower or "forecast" in c_lower:
        args: Dict[str, Any] = {}

        loc = _parse_location_after_in(c)
        if loc:
            args["location"] = loc

        days = _extract_days(c)
        if days is not None:
            args["days"] = max(1, min(14, days))
        elif "tomorrow" in c_lower:
            # Include today+tomorrow; tool returns current + daily list.
            args["days"] = 2
        elif "weekly" in c_lower or "week" in c_lower:
            args["days"] = 7

        if any(k in c_lower for k in ["fahrenheit", "imperial", " f ", " f."]):
            args["units"] = "fahrenheit"
        elif any(k in c_lower for k in ["celsius", "metric", " c ", " c."]):
            args["units"] = "celsius"

        return [Intent(tool="get_weather_forecast", args=args)]

    # --- Library refresh ---
    if "refresh library" in c_lower or "rebuild library" in c_lower or c_lower == "local library refresh":
        return [Intent(tool="local_library_refresh", args={})]

    # --- Audio output device switching (explicit only) ---
    dev = _parse_audio_device_target(c)
    if dev:
        return [Intent(tool="set_audio_output_device", args={"device": dev})]

    # --- Media / volume ---
    if any(k in c_lower for k in ["mute", "unmute"]):
        return [Intent(tool="volume_mute_toggle", args={})]

    if any(k in c_lower for k in ["volume up", "turn up", "louder"]):
        steps = _extract_int(c_lower)
        args: Dict[str, Any] = {}
        if steps is not None:
            args["steps"] = max(1, min(10, steps))
        return [Intent(tool="volume_up", args=args)]

    if any(k in c_lower for k in ["volume down", "turn down", "quieter"]):
        steps = _extract_int(c_lower)
        args = {}
        if steps is not None:
            args["steps"] = max(1, min(10, steps))
        return [Intent(tool="volume_down", args=args)]

    if any(k in c_lower for k in ["next track", "next song", "skip", "next"]):
        return [Intent(tool="media_next", args={})]

    if any(k in c_lower for k in ["previous track", "previous song", "prev", "previous", "back track"]):
        return [Intent(tool="media_previous", args={})]

    if any(k in c_lower for k in ["pause", "play", "resume", "play pause", "play/pause"]):
        return [Intent(tool="media_play_pause", args={})]

    # --- Window management ---
    m = re.match(r"^(focus|activate|switch\s+to)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="focus_window", args={"process": target})]

    m = re.match(r"^minimize\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="minimize_window", args={"process": target})]

    m = re.match(r"^(maximize|fullscreen)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="maximize_window", args={"process": target})]

    m = re.match(r"^(close|exit|quit)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            # Force close is intentionally not supported in fast-path.
            return [Intent(tool="close_window", args={"process": target, "force": False})]

    m = re.match(
        r"^move\s+(?P<target>.+?)\s+to\s+(?:(?P<primary>primary)\s+monitor|monitor\s+(?P<mon>\d+)|(?P<mon2>\d+))(?P<rest>.*)$",
        c_lower,
    )
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if not target:
            return None

        if m.group("primary"):
            monitor: Any = "primary"
        else:
            mon_s = m.group("mon") or m.group("mon2")
            if not mon_s:
                return None
            try:
                monitor = int(mon_s)
            except Exception:
                return None

        rest = (m.group("rest") or "").strip()
        position = "maximize"
        if " left" in f" {rest} " or rest.startswith("left"):
            position = "left"
        elif " right" in f" {rest} " or rest.startswith("right"):
            position = "right"
        elif " center" in f" {rest} " or rest.startswith("center"):
            position = "center"
        elif any(k in rest for k in ["maximize", "full", "fullscreen"]):
            position = "maximize"

        return [
            Intent(
                tool="move_window_to_monitor",
                args={"process": target, "monitor": monitor, "position": position},
            )
        ]

    # --- Open / launch ---
    m = re.match(r"^(open|launch|start)\s+(?P<q>.+)$", c_lower)
    if m:
        query = _strip_trailing_punct(m.group("q")).strip()
        if not query:
            return None

        # If the user explicitly indicates web intent OR query looks like a URL/domain, open as website.
        if _user_explicitly_requested_website(c_raw) or _looks_like_url(query) or query.lower() in _FASTPATH_COMMON_WEBSITES:
            return [Intent(tool="open_website", args={"url": query})]

        # Default: open locally (apps/games/files/folders) via local library resolution.
        return [Intent(tool="open_target", args={"query": query})]

    # "go to X" is usually web.
    m = re.match(r"^(go\s+to)\s+(?P<q>.+)$", c_lower)
    if m:
        query = _strip_trailing_punct(m.group("q")).strip()
        if query:
            return [Intent(tool="open_website", args={"url": query})]

    return None


def _format_fastpath_reply(user_text: str, intents: List[Intent], execution_summary: ExecutionSummary) -> str:
    def format_info(tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        if tool == "get_time":
            t = result.get("time")
            return f"It is {t}." if t else "Here's the current time."

        if tool == "get_system_info":
            os_name = result.get("os")
            arch = result.get("architecture")
            cores = result.get("cpu_cores")
            if os_name and arch and isinstance(cores, int):
                return f"{os_name} ({arch}), {cores} CPU cores."
            return "Here's your system information."

        if tool == "monitor_info":
            count = result.get("count")
            if isinstance(count, int):
                return f"You have {count} monitor(s)."
            return "Here's your monitor information."

        if tool == "get_location":
            city = result.get("city")
            region = result.get("region")
            country = result.get("country")
            parts = [p for p in [city, region, country] if isinstance(p, str) and p.strip()]
            if parts:
                return "You're in " + ", ".join(parts) + "."
            return "Here's your approximate location."

        if tool == "get_weather_forecast":
            loc = (result.get("location") or {}).get("name") if isinstance(result.get("location"), dict) else None
            current = result.get("current") if isinstance(result.get("current"), dict) else {}
            temp = current.get("temperature")
            weather = current.get("weather")
            if loc and (temp is not None or weather):
                if temp is not None and weather:
                    return f"{weather}, {temp}° in {loc}."
                if temp is not None:
                    return f"It's {temp}° in {loc}."
                return f"{weather} in {loc}."
            return "Here's the forecast."

        return None

    # Prefer first failure.
    for r in execution_summary.ran:
        if not r.ok:
            return "I couldn't complete that." if not r.error else f"I couldn't complete that: {r.error}"

    if not execution_summary.ran:
        return "Done."

    # Single-intent.
    if len(intents) == 1 and len(execution_summary.ran) == 1:
        tool = intents[0].tool
        args = intents[0].args or {}
        result = execution_summary.ran[0].result or {}

        info = format_info(tool, args, result)
        if info:
            return info

        if tool == "open_target":
            q = args.get("query")
            return f"Opening {q}." if q else "Opening."

        if tool == "open_website":
            url = args.get("url")
            return f"Opening {url}." if url else "Opening."

        if tool == "set_audio_output_device":
            chosen = result.get("chosen") if isinstance(result, dict) else None
            chosen_name = (chosen or {}).get("name") if isinstance(chosen, dict) else None
            if isinstance(chosen_name, str) and chosen_name.strip():
                return f"Audio output set to {chosen_name}."
            requested = args.get("device")
            return f"Switching audio output to {requested}." if requested else "Switching audio output."

        if tool in {"media_play_pause", "media_next", "media_previous", "volume_up", "volume_down", "volume_mute_toggle"}:
            return "OK."

        return "Done."

    # Multi-intent: summarize key actions + include the last info response (if any).
    opened: List[str] = []
    audio_switched: Optional[str] = None
    info_sentence: Optional[str] = None

    for idx, intent in enumerate(intents[: len(execution_summary.ran)]):
        res = execution_summary.ran[idx].result or {}
        args = intent.args or {}

        info = format_info(intent.tool, args, res)
        if info:
            info_sentence = info
            continue

        if intent.tool == "open_target":
            q = args.get("query")
            if isinstance(q, str) and q.strip():
                opened.append(q.strip())
            continue

        if intent.tool == "open_website":
            url = args.get("url")
            if isinstance(url, str) and url.strip():
                opened.append(url.strip())
            continue

        if intent.tool == "set_audio_output_device":
            chosen = res.get("chosen") if isinstance(res, dict) else None
            chosen_name = (chosen or {}).get("name") if isinstance(chosen, dict) else None
            if isinstance(chosen_name, str) and chosen_name.strip():
                audio_switched = chosen_name.strip()
            else:
                requested = args.get("device")
                if isinstance(requested, str) and requested.strip():
                    audio_switched = requested.strip()

    action_fragments: List[str] = []
    if opened:
        unique_opened: List[str] = []
        seen: set[str] = set()
        for o in opened:
            key = o.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_opened.append(o)

        if len(unique_opened) == 1:
            action_fragments.append(f"Opened {unique_opened[0]}")
        elif len(unique_opened) == 2:
            action_fragments.append(f"Opened {unique_opened[0]} and {unique_opened[1]}")
        else:
            action_fragments.append("Opened " + ", ".join(unique_opened[:-1]) + f", and {unique_opened[-1]}")

    if audio_switched:
        action_fragments.append(f"set audio output to {audio_switched}")

    action_sentence: Optional[str] = None
    if action_fragments:
        action_sentence = "; ".join(action_fragments) + "."

    if action_sentence and info_sentence:
        return f"{action_sentence} {info_sentence}"
    if info_sentence:
        return info_sentence
    if action_sentence:
        return action_sentence
    return "Done."


def _user_explicitly_requested_website(text: str) -> bool:
    t = (text or "").lower()
    # Keep this conservative to avoid breaking "open youtube" patterns.
    # Only treat as explicit web intent when the user says so.
    explicit_markers = [
        "website",
        "site",
        "web site",
        "web",
        "url",
        "link",
        "http://",
        "https://",
        "www.",
        ".com",
        ".net",
        ".org",
        ".io",
        "dot com",
    ]
    return any(m in t for m in explicit_markers)


def _extract_host_base(url: str) -> Optional[str]:
    if not url:
        return None

    raw = url.strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").split(":")[0].strip().lower()
    if not host:
        return None

    # Drop common prefixes
    if host.startswith("www."):
        host = host[4:]

    # Use the left-most label as base (rocketleague.com -> rocketleague)
    base = host.split(".")[0]
    base_norm = _normalize_alnum(base)
    return base_norm or None


def _find_matching_phrase_in_text(text: str, target_norm: str, max_ngram: int = 6) -> Optional[str]:
    """Find a phrase in the user text whose normalized form matches target_norm."""
    if not text or not target_norm:
        return None

    # Tokenize while preserving original tokens for reconstruction
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return None

    # Try longer n-grams first (more specific)
    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            phrase_norm = _normalize_alnum(phrase)
            if phrase_norm == target_norm:
                return phrase

    return None


def _rewrite_open_website_intents(user_text: str, intents) -> None:
    """Rewrite certain open_website intents to open_target when they likely refer to an installed app/game."""
    if not intents:
        return

    if _user_explicitly_requested_website(user_text):
        return

    for intent in intents:
        if intent.tool != "open_website":
            continue

        url = (intent.args or {}).get("url", "")
        base_norm = _extract_host_base(url)
        if not base_norm:
            continue

        phrase = _find_matching_phrase_in_text(user_text, base_norm)
        if not phrase:
            continue

        try:
            resolved = resolve_target(phrase)
        except Exception:
            continue

        r_type = resolved.get("type")
        confidence = float(resolved.get("confidence", 0) or 0)

        # Only rewrite when we're pretty confident it's a local thing.
        if r_type in {"game", "app", "uwp", "folder", "file"} and confidence >= 0.6:
            intent.tool = "open_target"
            intent.args = {"query": phrase}


def _execute_tool(registry, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool with validation and logging"""
    logger = get_logger_instance()
    
    # Check tool exists
    tool = registry.get(tool_name)
    if tool is None:
        return {
            "error": {
                "type": "tool_not_found",
                "message": f"Tool '{tool_name}' does not exist"
            }
        }
    
    # Validate arguments
    is_valid, error = validate_args(tool.args_schema, tool_args)
    if not is_valid:
        return {"error": error}
    
    # Log BEFORE execution
    logger.info(f"[TOOLS] Executing {tool_name} args={tool_args}")
    
    # Execute tool
    try:
        result = tool.run(**tool_args)
        
        # Log AFTER execution
        logger.info(f"[TOOLS] Result {result}")
        
        return result
    except Exception as e:
        error_result = {
            "error": {
                "type": "execution_error",
                "message": str(e)
            }
        }
        logger.info(f"[TOOLS] Result {error_result}")
        return error_result


def _call_llm(user_text: str, registry) -> Dict[str, Any]:
    """
    Call LLM for initial intent interpretation.
    
    Returns:
        Dict with either {"reply": "..."} or {"intents": [...]} or legacy formats
    """
    # Build tool list for prompt
    tools_list = registry.list_tools()
    tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
    
    prompt = f"""You are Wyzer, a local voice assistant. You can use tools to help users.

Available tools:
{tools_desc}

TOOL USAGE GUIDANCE:
- For "open X" requests:
    - If X is an installed app/game/folder/file name, use open_target with query=X.
    - Use open_website ONLY when the user explicitly requests a website/URL (e.g., says "website", provides a domain like "example.com", or says "go to ...").
    - Do NOT invent URLs for non-web apps/games. (Example: "open Rocket League" is a game -> open_target, not open_website.)
- For window control: use focus_window, minimize_window, maximize_window, close_window, or move_window_to_monitor
  - move_window_to_monitor: Use monitor="primary" for primary monitor, or monitor=0/1/2 for specific monitor index
- For media control: use media_play_pause, media_next, media_previous for playback; volume_up, volume_down, volume_mute_toggle for audio
- For switching the default audio output device (speakers/headset): use set_audio_output_device with device="name" (fuzzy match allowed)
- For monitor info: use monitor_info to check available monitors
- For library management: use local_library_refresh to rebuild the index
- For location: use get_location (approximate IP-based)
- For weather/forecast: use get_weather_forecast (optionally pass location; otherwise it uses IP-based location)
    - If user asks for Fahrenheit, pass units="fahrenheit" (or units="imperial")

MULTI-INTENT SUPPORT (NEW):
- You can now execute MULTIPLE tools in sequence for complex requests
- Use "intents" array to specify multiple actions in order
- Each intent has: {{"tool": "tool_name", "args": {{...}}, "continue_on_error": false}}
- Keep intents under 5 per request
- Preserve order - they execute sequentially

EXAMPLES:
User: "open downloads"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "downloads"}}}}], "reply": "Opening downloads"}}

User: "launch chrome and open youtube"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "chrome"}}}}, {{"tool": "open_website", "args": {{"url": "youtube"}}}}], "reply": "Launching Chrome and opening YouTube"}}

User: "open Rocket League"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "Rocket League"}}}}], "reply": "Opening Rocket League"}}

User: "open rocketleague.com"
Response: {{"intents": [{{"tool": "open_website", "args": {{"url": "rocketleague.com"}}}}], "reply": "Opening rocketleague.com"}}

User: "pause music and mute volume"
Response: {{"intents": [{{"tool": "media_play_pause", "args": {{}}}}, {{"tool": "volume_mute_toggle", "args": {{}}}}], "reply": "Pausing and muting"}}

User: "move chrome to primary monitor"
Response: {{"intents": [{{"tool": "move_window_to_monitor", "args": {{"process": "chrome", "monitor": "primary"}}}}], "reply": "Moving Chrome to your primary monitor"}}

User: "what time is it"
Response: {{"intents": [{{"tool": "get_time", "args": {{}}}}], "reply": ""}}

User: "hello"
Response: {{"reply": "Hello! How can I help you?"}}

RESPONSE FORMAT: You must respond with valid JSON only (no markdown, no code blocks).
Option 1 - Direct reply (no tool needed):
{{"reply": "your response here"}}

Option 2 - Single tool:
{{"intents": [{{"tool": "tool_name", "args": {{"key": "value"}}}}], "reply": "brief message"}}

Option 3 - Multiple tools (for multi-step requests):
{{"intents": [{{"tool": "tool1", "args": {{}}}}, {{"tool": "tool2", "args": {{}}}}], "reply": "brief message"}}

Rules:
- Default to 1-2 sentences.
- If the user explicitly asks for a long story, an in-depth explanation, step-by-step details, or to "go into detail", you may write a longer response.
- Be direct and helpful
- Use intents when appropriate to answer the user's question
- Preserve order for multi-step actions
- If using tools, I will run them and ask you again for the final reply

User: {user_text}

Your response (JSON only):"""

    return _ollama_request(prompt)


def _call_llm_with_execution_summary(
    user_text: str,
    execution_summary: ExecutionSummary,
    registry
) -> Dict[str, Any]:
    """
    Call LLM after tool execution(s) to generate final reply.
    
    Args:
        user_text: Original user request
        execution_summary: Summary of executed intents
        registry: Tool registry
        
    Returns:
        Dict with {"reply": "..."}
    """
    # Build a summary of what was executed
    summary_parts = []
    for result in execution_summary.ran:
        if result.ok:
            summary_parts.append(
                f"- Executed '{result.tool}' successfully. Result: {json.dumps(result.result)}"
            )
        else:
            summary_parts.append(
                f"- Failed to execute '{result.tool}'. Error: {result.error}"
            )
    
    if execution_summary.stopped_early:
        summary_parts.append("- Execution stopped early due to error")
    
    summary_text = "\n".join(summary_parts)
    
    prompt = f"""You are Wyzer, a local voice assistant.

The user asked: {user_text}

I executed the following actions:
{summary_text}

Now provide a natural reply to the user based on these results.
- Default to 1-2 sentences.
- If the user's original request asked for detail/step-by-step/a long explanation or a story, provide a longer, more in-depth reply.

RESPONSE FORMAT: JSON only (no markdown):
{{"reply": "your natural response to the user"}}

Your response (JSON only):"""

    response = _ollama_request(prompt)
    return response


def _call_llm_with_tool_result(
    user_text: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Dict[str, Any],
    registry
) -> Dict[str, Any]:
    """
    Call LLM after tool execution to generate final reply.
    
    Returns:
        Dict with {"reply": "..."}
    """
    prompt = f"""You are Wyzer, a local voice assistant.

The user asked: {user_text}

I executed the tool '{tool_name}' with arguments: {json.dumps(tool_args)}

Tool result: {json.dumps(tool_result)}

Now provide a natural reply to the user based on this result.
- Default to 1-2 sentences.
- If the user's original request asked for detail/step-by-step/a long explanation or a story, provide a longer, more in-depth reply.

RESPONSE FORMAT: JSON only (no markdown):
{{"reply": "your natural response to the user"}}

Your response (JSON only):"""

    response = _ollama_request(prompt)
    return response


def _ollama_request(prompt: str) -> Dict[str, Any]:
    """
    Make request to Ollama LLM.
    
    Returns:
        Parsed JSON response or fallback dict
    """
    logger = get_logger_instance()
    try:
        # Use existing config
        base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        model = Config.OLLAMA_MODEL
        timeout = Config.LLM_TIMEOUT
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Request JSON output
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 150
            }
        }
        
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
        
        # Parse response
        reply_text = response_data.get("response", "").strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(reply_text)
            return parsed
        except json.JSONDecodeError:
            # LLM didn't return valid JSON, extract reply if possible
            return {"reply": reply_text if reply_text else "I couldn't process that."}

    except urllib.error.URLError as e:
        # Distinguish slow-model timeouts from true connection failures.
        reason = getattr(e, "reason", None)
        is_timeout = isinstance(reason, socket.timeout) or "timed out" in str(e).lower()
        if is_timeout:
            logger.warning(f"Ollama request timed out after {Config.LLM_TIMEOUT}s: {e}")
            return {
                "reply": f"Ollama is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
            }

        logger.warning(f"Ollama request failed (URL error): {e}")
        return {"reply": "I couldn't reach Ollama. Is it running?"}

    except socket.timeout as e:
        logger.warning(f"Ollama request timed out after {Config.LLM_TIMEOUT}s: {e}")
        return {
            "reply": f"Ollama is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
        }

    except Exception as e:
        # Keep generic fallback, but log the underlying error for debugging.
        logger.exception(f"Unexpected Ollama error: {e}")
        return {"reply": "I had trouble talking to Ollama."}
