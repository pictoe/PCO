"""
Open target tool - opens folders, files, apps, or URLs based on user query.
"""
import os
import time
import subprocess
from typing import Dict, Any
from pathlib import Path
from wyzer.tools.tool_base import ToolBase
from wyzer.local_library import resolve_target


class OpenTargetTool(ToolBase):
    """Tool to open folders, files, apps, or URLs"""
    
    def __init__(self):
        """Initialize open_target tool"""
        super().__init__()
        self._name = "open_target"
        self._description = "Open a folder, file, app, or URL based on natural language query (e.g., 'downloads', 'chrome', 'my documents')"
        self._args_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Natural language query for what to open (e.g., 'downloads', 'notepad', 'my pictures')"
                },
                "open_mode": {
                    "type": "string",
                    "enum": ["default", "folder", "file", "app"],
                    "description": "Optional: force specific open mode"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Open a target based on query.
        
        Args:
            query: Natural language query
            open_mode: Optional mode override
            
        Returns:
            Dict with status and resolved info, or error
        """
        start_time = time.perf_counter()
        
        query = kwargs.get("query", "").strip()
        open_mode = kwargs.get("open_mode", "default")
        
        if not query:
            return {
                "error": {
                    "type": "invalid_query",
                    "message": "Query cannot be empty"
                }
            }
        
        try:
            # Resolve target
            resolved = resolve_target(query)
            
            if resolved.get("type") == "unknown":
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "not_found",
                        "message": f"Could not find a match for '{query}'"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            # Check confidence threshold
            if resolved.get("confidence", 0) < 0.3:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "low_confidence",
                        "message": f"Low confidence match for '{query}'"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            # Open based on type
            target_type = resolved.get("type")
            target_path = resolved.get("path", "")
            
            if target_type == "url":
                # Delegate to open_website (use webbrowser)
                import webbrowser
                url = resolved.get("url", query)
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                webbrowser.open(url)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "game":
                # Launch game based on launch type
                launch_info = resolved.get("launch", {})
                launch_type = launch_info.get("type", "")
                launch_target = launch_info.get("target", "")
                
                try:
                    if launch_type == "steam_uri":
                        # Open Steam URI
                        import webbrowser
                        webbrowser.open(launch_target)
                    
                    elif launch_type == "epic_uri":
                        # Open Epic Games launcher URI
                        import webbrowser
                        webbrowser.open(launch_target)
                    
                    elif launch_type == "exe":
                        # Launch executable directly
                        subprocess.Popen([launch_target], shell=False)
                    
                    elif launch_type == "shortcut":
                        # Open .lnk shortcut
                        os.startfile(launch_target)
                    
                    elif launch_type == "uwp":
                        # Launch UWP app via explorer
                        subprocess.run(["explorer", launch_target], check=False)
                    
                    else:
                        end_time = time.perf_counter()
                        latency_ms = int((end_time - start_time) * 1000)
                        
                        return {
                            "error": {
                                "type": "unsupported_launch_type",
                                "message": f"Unsupported game launch type: {launch_type}"
                            },
                            "resolved": resolved,
                            "latency_ms": latency_ms
                        }
                    
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "status": "opened",
                        "resolved": resolved,
                        "game_name": resolved.get("game_name", ""),
                        "latency_ms": latency_ms
                    }
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "error": {
                            "type": "launch_error",
                            "message": f"Failed to launch game: {str(e)}"
                        },
                        "resolved": resolved,
                        "latency_ms": latency_ms
                    }
            
            elif target_type == "uwp":
                # Launch UWP app via explorer shell:AppsFolder
                app_id = resolved.get("path", "")
                
                try:
                    subprocess.Popen(
                        ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                        shell=False
                    )
                    
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "status": "opened",
                        "resolved": resolved,
                        "app_name": resolved.get("app_name", ""),
                        "latency_ms": latency_ms
                    }
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "error": {
                            "type": "launch_error",
                            "message": f"Failed to launch UWP app: {str(e)}"
                        },
                        "resolved": resolved,
                        "latency_ms": latency_ms
                    }
            
            elif target_type == "folder":
                # Open folder in Explorer
                self._open_folder(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "file":
                # Open file (select in Explorer or open with default app)
                if open_mode == "folder":
                    # Open parent folder with file selected
                    self._open_file_location(target_path)
                else:
                    # Open with default app
                    os.startfile(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "app":
                # Launch app
                self._launch_app(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            else:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "unsupported_type",
                        "message": f"Unsupported target type: {target_type}"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
                
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
    
    def _open_folder(self, path: str) -> None:
        """Open folder in Windows Explorer"""
        os.startfile(path)
    
    def _open_file_location(self, path: str) -> None:
        """Open file location in Windows Explorer with file selected"""
        subprocess.run(["explorer", "/select,", path], check=False)
    
    def _launch_app(self, path: str) -> None:
        """Launch an application"""
        # Check if it's a .lnk shortcut
        if path.lower().endswith(".lnk"):
            os.startfile(path)
        else:
            # Launch executable
            subprocess.Popen([path], shell=False)
