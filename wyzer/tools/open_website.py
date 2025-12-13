"""
Open website tool.
"""
import webbrowser
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class OpenWebsiteTool(ToolBase):
    """Tool to open a website in the default browser"""
    
    def __init__(self):
        """Initialize open_website tool"""
        super().__init__()
        self._name = "open_website"
        self._description = "Open a website URL in the default browser"
        self._args_schema = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Full URL starting with http:// or https://"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Open a website URL.
        
        Args:
            url: Website URL to open
            
        Returns:
            Dict with status and url, or error
        """
        url = kwargs.get("url", "")
        
        # Validate URL format
        if not url.startswith("http://") and not url.startswith("https://"):
            return {
                "error": {
                    "type": "invalid_url",
                    "message": "URL must start with http:// or https://"
                }
            }
        
        try:
            # Open URL in default browser
            webbrowser.open(url)
            return {
                "status": "opened",
                "url": url
            }
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
