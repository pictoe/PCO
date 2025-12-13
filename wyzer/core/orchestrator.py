"""
Orchestrator for Wyzer AI Assistant - Phase 6
Coordinates LLM reasoning and tool execution.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any
from wyzer.core.config import Config
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args

# Module-level singleton registry
_registry = None


def get_registry():
    """Get or create the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def handle_user_text(text: str) -> Dict[str, str]:
    """
    Handle user text input with optional tool execution.
    
    Args:
        text: User's input text
        
    Returns:
        Dict with "reply" key containing the final response
    """
    try:
        registry = get_registry()
        
        # First LLM call: interpret user intent
        llm_response = _call_llm(text, registry)
        
        # Check if LLM wants to use a tool
        if "tool" in llm_response and llm_response.get("tool"):
            tool_name = llm_response["tool"]
            tool_args = llm_response.get("args", {})
            
            # Execute tool
            tool_result = _execute_tool(registry, tool_name, tool_args)
            
            # Second LLM call: generate final reply with tool result
            final_response = _call_llm_with_tool_result(
                text, tool_name, tool_args, tool_result, registry
            )
            return {"reply": final_response.get("reply", "I executed the action.")}
        else:
            # No tool needed, return direct reply
            return {"reply": llm_response.get("reply", "")}
            
    except Exception as e:
        # Safe fallback on any error
        return {"reply": f"I encountered an error: {str(e)}"}


def _execute_tool(registry, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool with validation"""
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
    
    # Execute tool
    try:
        result = tool.run(**tool_args)
        return result
    except Exception as e:
        return {
            "error": {
                "type": "execution_error",
                "message": str(e)
            }
        }


def _call_llm(user_text: str, registry) -> Dict[str, Any]:
    """
    Call LLM for initial intent interpretation.
    
    Returns:
        Dict with either {"reply": "..."} or {"tool": "...", "args": {...}}
    """
    # Build tool list for prompt
    tools_list = registry.list_tools()
    tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
    
    prompt = f"""You are Wyzer, a local voice assistant. You can use tools to help users.

Available tools:
{tools_desc}

RESPONSE FORMAT: You must respond with valid JSON only (no markdown, no code blocks).
Option 1 - Direct reply (no tool needed):
{{"reply": "your response here"}}

Option 2 - Use a tool:
{{"tool": "tool_name", "args": {{"key": "value"}}, "reply": "optional brief message"}}

Rules:
- Keep replies to 1-2 sentences
- Be direct and helpful
- Use tools when appropriate to answer the user's question
- If using a tool, I will run it and ask you again for the final reply

User: {user_text}

Your response (JSON only):"""

    return _ollama_request(prompt)


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
