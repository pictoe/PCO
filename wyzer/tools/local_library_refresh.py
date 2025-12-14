"""
Local library refresh tool.
"""
import time
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase
from wyzer.local_library import refresh_index


class LocalLibraryRefreshTool(ToolBase):
    """Tool to refresh the local library index"""
    
    def __init__(self):
        """Initialize local_library_refresh tool"""
        super().__init__()
        self._name = "local_library_refresh"
        self._description = "Refresh the index of local folders, apps, and shortcuts (use when user wants to update or rebuild the library)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Refresh the local library index.
        
        Returns:
            Dict with status and counts, or error
        """
        start_time = time.perf_counter()
        
        try:
            result = refresh_index()
            
            # Add latency if not present
            if "latency_ms" not in result:
                end_time = time.perf_counter()
                result["latency_ms"] = int((end_time - start_time) * 1000)
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "refresh_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
