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
        self._description = "Open a website in the default browser (use for explicit websites/URLs like 'youtube', 'github.com', 'rocketleague.com', or full https:// URLs)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Website name or URL (e.g., 'facebook', 'youtube.com', or 'https://example.com')"
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
        url = kwargs.get("url", "").strip()
        
        if not url:
            return {
                "error": {
                    "type": "invalid_url",
                    "message": "URL cannot be empty"
                }
            }
        
        # Normalize URL: add https:// if no scheme present
        if not url.startswith("http://") and not url.startswith("https://"):
            # Check if it looks like a valid domain or common shorthand
            # Accept formats like: "facebook", "facebook.com", "openfacebook.com", "youtube"
            if "." not in url:
                # Simple name like "facebook" or "youtube" -> add .com
                url = f"https://{url}.com"
            else:
                # Has dot, assume it's a domain like "facebook.com"
                url = f"https://{url}"
        
        # Basic validation: reject obviously invalid URLs
        if " " in url or url.count("://") > 1:
            return {
                "error": {
                    "type": "invalid_url",
                    "message": "URL format is invalid"
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
