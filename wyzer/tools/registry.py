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

    # Location / Weather tools
    from wyzer.tools.get_location import GetLocationTool
    from wyzer.tools.get_weather_forecast import GetWeatherForecastTool
    
    # Phase 6 tools - LocalLibrary
    from wyzer.tools.local_library_refresh import LocalLibraryRefreshTool
    from wyzer.tools.open_target import OpenTargetTool
    
    # Phase 6 tools - Window management
    from wyzer.tools.window_manager import (
        FocusWindowTool,
        MinimizeWindowTool,
        MaximizeWindowTool,
        CloseWindowTool,
        MoveWindowToMonitorTool
    )
    
    # Phase 6 tools - Monitor info
    from wyzer.tools.monitor_info import MonitorInfoTool
    
    # Phase 6 tools - Media controls
    from wyzer.tools.media_controls import (
        MediaPlayPauseTool,
        MediaNextTool,
        MediaPreviousTool,
        VolumeUpTool,
        VolumeDownTool,
        VolumeMuteToggleTool
    )

    # True volume control (pycaw)
    from wyzer.tools.volume_control import VolumeControlTool

    # Phase 6 tools - Audio device switching
    from wyzer.tools.audio_output_device import SetAudioOutputDeviceTool
    
    registry = ToolRegistry()
    
    # Register default tools
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    registry.register(OpenWebsiteTool())

    # Register location/weather tools
    registry.register(GetLocationTool())
    registry.register(GetWeatherForecastTool())
    
    # Register LocalLibrary tools
    registry.register(LocalLibraryRefreshTool())
    registry.register(OpenTargetTool())
    
    # Register window management tools
    registry.register(FocusWindowTool())
    registry.register(MinimizeWindowTool())
    registry.register(MaximizeWindowTool())
    registry.register(CloseWindowTool())
    registry.register(MoveWindowToMonitorTool())
    
    # Register monitor info tool
    registry.register(MonitorInfoTool())
    
    # Register media control tools
    registry.register(MediaPlayPauseTool())
    registry.register(MediaNextTool())
    registry.register(MediaPreviousTool())
    registry.register(VolumeUpTool())
    registry.register(VolumeDownTool())
    registry.register(VolumeMuteToggleTool())

    # Register true volume control tool
    registry.register(VolumeControlTool())

    # Register audio device switching tool
    registry.register(SetAudioOutputDeviceTool())
    
    return registry
