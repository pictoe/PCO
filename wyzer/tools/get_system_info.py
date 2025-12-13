"""
Get system information tool.
"""
import os
import platform
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class GetSystemInfoTool(ToolBase):
    """Tool to get system information"""
    
    def __init__(self):
        """Initialize get_system_info tool"""
        super().__init__()
        self._name = "get_system_info"
        self._description = "Get basic system information (OS, CPU, RAM)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dict with os, cpu_cores, ram_gb (if available)
        """
        try:
            result = {
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine()
            }
            
            # Try to get CPU cores
            try:
                result["cpu_cores"] = os.cpu_count() or 0
            except:
                result["cpu_cores"] = 0
            
            # Try psutil for RAM info (optional)
            try:
                import psutil
                mem = psutil.virtual_memory()
                result["ram_gb"] = round(mem.total / (1024**3), 2)
                result["ram_available_gb"] = round(mem.available / (1024**3), 2)
            except ImportError:
                # psutil not available, that's okay
                result["ram_gb"] = None
            
            return result
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
