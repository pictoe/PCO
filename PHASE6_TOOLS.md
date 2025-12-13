# Wyzer Phase 6 - Tools System

## Overview

Phase 6 adds a clean orchestrator-based tools system that allows the LLM to execute actions via a controlled, stateless tool API.

## Architecture

```
User Speech → STT → Orchestrator → LLM (interprets intent)
                         ↓
                    Tool requested?
                         ↓
                    Validate & Execute Tool
                         ↓
                    LLM (format final reply) → TTS → User
```

## Key Principles

1. **LLM never executes tools directly** - Orchestrator validates and runs tools
2. **Tools are stateless** - Return JSON dicts only (no side effects on state)
3. **Strict JSON protocol** - LLM outputs structured JSON for parsing
4. **Safe validation** - Arguments validated against schemas before execution
5. **Minimal assistant.py changes** - Audio pipeline unchanged

## Files Created

### Core
- `wyzer/orchestrator.py` - Main orchestrator logic
- `wyzer/tools/__init__.py` - Tools package

### Tools Framework
- `wyzer/tools/tool_base.py` - ToolBase abstract class
- `wyzer/tools/registry.py` - Tool registration and discovery
- `wyzer/tools/validation.py` - Argument validation (JSON-schema-like)

### Default Tools
- `wyzer/tools/get_time.py` - Get current time
- `wyzer/tools/get_system_info.py` - Get OS/CPU/RAM info
- `wyzer/tools/open_website.py` - Open URL in browser

## LLM Protocol

The LLM must respond with valid JSON (no markdown):

**Option 1 - Direct reply (no tool):**
```json
{"reply": "Sure, I can help with that."}
```

**Option 2 - Request tool:**
```json
{
  "tool": "get_time",
  "args": {},
  "reply": "Let me check that for you"
}
```

When a tool is requested:
1. Orchestrator validates tool exists
2. Validates arguments against schema
3. Executes tool → gets result JSON
4. Calls LLM again with context (user query + tool result)
5. LLM formats natural response
6. Returns `{"reply": "It's 3:45 PM"}`

## Testing

### Test Mode
```bash
set WYZER_TOOLS_TEST=1
python run.py
```

This runs quick tests without starting the full assistant:
- "what time is it"
- "what's the system information"
- "open https://example.com"

### Normal Mode
```bash
python run.py
```

Tools work automatically when users ask relevant questions.

## Adding New Tools

1. Create `wyzer/tools/my_tool.py`:
```python
from wyzer.tools.tool_base import ToolBase

class MyTool(ToolBase):
    def __init__(self):
        super().__init__()
        self._name = "my_tool"
        self._description = "What this tool does"
        self._args_schema = {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        }
    
    def run(self, **kwargs):
        # Do work (stateless)
        return {"result": "success"}
```

2. Register in `wyzer/tools/registry.py`:
```python
from wyzer.tools.my_tool import MyTool

def build_default_registry():
    registry = ToolRegistry()
    registry.register(MyTool())  # Add this
    return registry
```

## Integration Points

### assistant.py
Single change in `_background_think()`:
```python
from wyzer.core.orchestrator import handle_user_text
result_dict = handle_user_text(transcript)
```

Everything else unchanged - audio, hotword, TTS all work the same.

## Security

- **No arbitrary code execution** - Tools are fixed allowlist
- **URL validation** - `open_website` only accepts http(s)://
- **Argument validation** - Schema-based checks before execution
- **Error handling** - Graceful fallbacks, no exception leaks

## Configuration

Uses existing `wyzer/core/config.py`:
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `LLM_TIMEOUT`

No new config needed.
