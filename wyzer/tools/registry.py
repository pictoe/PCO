"""
Tool registry for managing available tools.
"""
from typing import Dict, List, Optional
from wyzer.tools.tool_base import ToolBase


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, ToolBase] = {}
    
    def register(self, tool: ToolBase) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolBase]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all registered tools with their metadata.
        
        Returns:
            List of dicts with name and description
        """
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self._tools.values()
        ]
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self._tools


def build_default_registry() -> ToolRegistry:
    """
    Build registry with default tools.
    
    Returns:
        ToolRegistry with standard tools registered
    """
    from wyzer.tools.get_time import GetTimeTool
    from wyzer.tools.get_system_info import GetSystemInfoTool
    from wyzer.tools.open_website import OpenWebsiteTool
    
    registry = ToolRegistry()
    
    # Register default tools
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    registry.register(OpenWebsiteTool())
    
    return registry
