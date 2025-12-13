"""
UWP app indexer for LocalLibrary - discovers Windows Store/UWP applications.
"""
import json
import subprocess
from typing import Dict, Any, List


def refresh_uwp_index() -> Dict[str, Any]:
    """
    Refresh UWP apps index using PowerShell Get-StartApps.
    
    Returns:
        {
            "status": "ok",
            "count": int,
            "apps": [...]
        }
        or
        {
            "error": {"type": str, "message": str},
            "apps": [],
            "count": 0
        }
    """
    apps = []
    
    try:
        # Use Get-StartApps which returns all Start Menu apps including UWP
        ps_command = "Get-StartApps | ConvertTo-Json -Compress"
        
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode != 0:
            return {
                "error": {
                    "type": "powershell_error",
                    "message": f"PowerShell command failed with exit code {result.returncode}"
                },
                "apps": [],
                "count": 0
            }
        
        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            return {
                "error": {
                    "type": "empty_output",
                    "message": "PowerShell returned empty output"
                },
                "apps": [],
                "count": 0
            }
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return {
                "error": {
                    "type": "json_parse_error",
                    "message": f"Failed to parse PowerShell JSON output: {str(e)}"
                },
                "apps": [],
                "count": 0
            }
        
        # Handle single object vs list
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return {
                "error": {
                    "type": "invalid_data_type",
                    "message": f"Expected list or dict, got {type(data).__name__}"
                },
                "apps": [],
                "count": 0
            }
        
        # Process each app
        for item in data:
            name = item.get("Name", "")
            app_id = item.get("AppID", "")
            
            if not name or not app_id:
                continue
            
            # Filter out non-UWP apps (traditional desktop apps)
            # UWP apps have AppID in format: CompanyName.AppName_hash!AppName
            if "!" not in app_id:
                continue
            
            apps.append({
                "name": name,
                "app_id": app_id,
                "source": "uwp",
                "aliases": _generate_uwp_aliases(name)
            })
        
        return {
            "status": "ok",
            "count": len(apps),
            "apps": apps
        }
        
    except subprocess.TimeoutExpired:
        return {
            "error": {
                "type": "timeout",
                "message": "PowerShell command timed out after 15 seconds"
            },
            "apps": [],
            "count": 0
        }
    except Exception as e:
        return {
            "error": {
                "type": "unexpected_error",
                "message": str(e)
            },
            "apps": [],
            "count": 0
        }


def _generate_uwp_aliases(name: str) -> List[str]:
    """
    Generate simple aliases for a UWP app name.
    
    Args:
        name: App display name
        
    Returns:
        List of alias strings (lowercase)
    """
    aliases = []
    
    # Base normalized name
    normalized = name.lower().strip()
    if len(normalized) >= 3:
        aliases.append(normalized)
    
    # Remove common suffix words
    words = normalized.split()
    suffix_words_to_remove = {"app", "desktop", "music", "launcher"}
    
    # Try removing last word if it's in suffix list
    if len(words) > 1 and words[-1] in suffix_words_to_remove:
        shortened = " ".join(words[:-1])
        if len(shortened) >= 3:
            aliases.append(shortened)
    
    # Add first word if long enough
    if len(words) > 1 and len(words[0]) >= 3:
        aliases.append(words[0])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        if alias not in seen and len(alias) >= 3:
            seen.add(alias)
            unique_aliases.append(alias)
    
    return unique_aliases


def load_uwp_index() -> Dict[str, Any]:
    """
    Load cached UWP apps from library.json.
    
    Returns:
        {"uwp_apps": [...], "uwp_scan_meta": {...}}
    """
    from wyzer.local_library.indexer import get_cached_index
    
    index = get_cached_index()
    return {
        "uwp_apps": index.get("uwp_apps", []),
        "uwp_scan_meta": index.get("uwp_scan_meta", {})
    }
