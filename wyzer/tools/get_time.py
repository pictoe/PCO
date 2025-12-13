"""
Get current time tool.
"""
from datetime import datetime
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class GetTimeTool(ToolBase):
    """Tool to get current time"""
    
    def __init__(self):
        """Initialize get_time tool"""
        super().__init__()
        self._name = "get_time"
        self._description = "Get the current local time"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get current time.
        
        Returns:
            Dict with time, timezone
        """
        try:
            now = datetime.now()
            return {
                "time": now.strftime("%H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "timezone": "local"
            }
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
