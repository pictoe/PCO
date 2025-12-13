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
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args
from wyzer.local_library import resolve_target
from wyzer.core.intent_plan import (
    normalize_plan,
    validate_intents,
    ExecutionResult,
    ExecutionSummary
)

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
- Keep replies to 1-2 sentences
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

Now provide a natural, concise reply to the user (1-2 sentences) based on these results.

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

Now provide a natural, concise reply to the user (1-2 sentences) based on this result.

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
            
    except Exception as e:
        # Fallback on any error
        return {"reply": "I couldn't connect to my brain. Is Ollama running?"}
