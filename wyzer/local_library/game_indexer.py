"""
Game indexer for LocalLibrary - discovers and indexes games from multiple sources.
"""
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


def refresh_games_index() -> Dict[str, Any]:
    """
    Refresh the games index from all available sources.
    
    Sources:
    - Steam (via registry + VDF parsing)
    - Epic Games (via manifest JSON files)
    - Shortcuts (Start Menu, Desktop)
    - Folder scan (common game directories)
    - Xbox/Microsoft Store (via PowerShell Get-AppxPackage)
    
    Returns:
        {
            "status": "ok",
            "counts": {"steam": int, "epic": int, ...},
            "sources": {"steam": bool, "epic": bool, ...},
            "errors": [...]  # optional
        }
    """
    start_time = time.perf_counter()
    
    games = []
    counts = {}
    sources = {}
    errors = []
    
    # 1) Index Steam games
    try:
        steam_games = _index_steam_games()
        games.extend(steam_games)
        counts["steam"] = len(steam_games)
        sources["steam"] = True
    except Exception as e:
        counts["steam"] = 0
        sources["steam"] = False
        errors.append({"source": "steam", "error": str(e)})
    
    # 2) Index Epic Games
    try:
        epic_games = _index_epic_games()
        games.extend(epic_games)
        counts["epic"] = len(epic_games)
        sources["epic"] = True
    except Exception as e:
        counts["epic"] = 0
        sources["epic"] = False
        errors.append({"source": "epic", "error": str(e)})
    
    # 3) Index game shortcuts (Start Menu + Desktop)
    try:
        shortcut_games = _index_game_shortcuts()
        games.extend(shortcut_games)
        counts["shortcuts"] = len(shortcut_games)
        sources["shortcuts"] = True
    except Exception as e:
        counts["shortcuts"] = 0
        sources["shortcuts"] = False
        errors.append({"source": "shortcuts", "error": str(e)})
    
    # 4) Folder scan fallback
    try:
        folder_games = _index_folder_scan()
        games.extend(folder_games)
        counts["folder_scan"] = len(folder_games)
        sources["folder_scan"] = True
    except Exception as e:
        counts["folder_scan"] = 0
        sources["folder_scan"] = False
        errors.append({"source": "folder_scan", "error": str(e)})
    
    # 5) Xbox/Microsoft Store (best effort)
    try:
        xbox_games = _index_xbox_games()
        games.extend(xbox_games)
        counts["xbox"] = len(xbox_games)
        sources["xbox"] = True
    except Exception as e:
        counts["xbox"] = 0
        sources["xbox"] = False
        errors.append({"source": "xbox", "error": str(e)})
    
    # Deduplicate games by name (prefer higher confidence sources)
    games = _deduplicate_games(games)
    
    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)
    
    result = {
        "status": "ok",
        "counts": counts,
        "sources": sources,
        "total_games": len(games),
        "games": games,
        "latency_ms": latency_ms
    }
    
    if errors:
        result["errors"] = errors
    
    return result


def load_games_index() -> Dict[str, Any]:
    """
    Load cached games index from library.json.
    
    Returns:
        Dict with games list, or empty structure if not found
    """
    from wyzer.local_library.indexer import LIBRARY_JSON_PATH, get_cached_index
    
    index = get_cached_index()
    return {
        "games": index.get("games", []),
        "games_scan_meta": index.get("games_scan_meta", {})
    }


def merge_games_into_library(library: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge refreshed games index into library data.
    
    Args:
        library: Existing library data dict
        
    Returns:
        Updated library data with games merged
    """
    games_result = refresh_games_index()
    
    # Merge games
    library["games"] = games_result["games"]
    
    # Update scan metadata
    if "games_scan_meta" not in library:
        library["games_scan_meta"] = {}
    
    library["games_scan_meta"]["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
    library["games_scan_meta"]["counts"] = games_result["counts"]
    library["games_scan_meta"]["sources"] = games_result["sources"]
    library["games_scan_meta"]["total_games"] = games_result["total_games"]
    
    if "errors" in games_result:
        library["games_scan_meta"]["errors"] = games_result["errors"]
    
    return library


# ==============================================================================
# STEAM INDEXER
# ==============================================================================

def _index_steam_games() -> List[Dict[str, Any]]:
    """
    Index Steam games from registry and VDF files.
    
    Returns:
        List of game dicts
    """
    games = []
    
    # Find Steam install path from registry
    steam_path = _get_steam_install_path()
    if not steam_path:
        return games
    
    # Find library folders
    library_folders = _parse_steam_library_folders(steam_path)
    
    # Parse appmanifest files in each library
    for library_path in library_folders:
        steamapps_path = Path(library_path) / "steamapps"
        if not steamapps_path.exists():
            continue
        
        try:
            for manifest_file in steamapps_path.glob("appmanifest_*.acf"):
                try:
                    game_data = _parse_steam_manifest(manifest_file)
                    if game_data:
                        # Build Steam URI
                        app_id = game_data.get("appid", "")
                        game_name = game_data.get("name", "")
                        install_dir = game_data.get("installdir", "")
                        
                        if app_id and game_name:
                            install_path = str(steamapps_path / "common" / install_dir) if install_dir else None
                            
                            games.append({
                                "name": game_name,
                                "source": "steam",
                                "launch": {
                                    "type": "steam_uri",
                                    "target": f"steam://rungameid/{app_id}"
                                },
                                "install_path": install_path,
                                "app_id": app_id,
                                "aliases": _generate_game_aliases(game_name),
                                "confidence": 0.95
                            })
                except Exception:
                    # Skip individual manifest errors
                    pass
        except Exception:
            # Skip library folder errors
            pass
    
    return games


def _get_steam_install_path() -> Optional[str]:
    """Get Steam installation path from Windows registry."""
    try:
        import winreg
        
        # Try HKEY_CURRENT_USER first
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam")
            steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
            winreg.CloseKey(key)
            return steam_path.replace("/", "\\")
        except:
            pass
        
        # Try HKEY_LOCAL_MACHINE
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam")
            install_path, _ = winreg.QueryValueEx(key, "InstallPath")
            winreg.CloseKey(key)
            return install_path
        except:
            pass
        
        # Fallback to common location
        common_path = Path(r"C:\Program Files (x86)\Steam")
        if common_path.exists():
            return str(common_path)
        
        return None
        
    except Exception:
        return None


def _parse_steam_library_folders(steam_path: str) -> List[str]:
    """
    Parse Steam's libraryfolders.vdf to get all library paths.
    
    Args:
        steam_path: Steam installation directory
        
    Returns:
        List of library folder paths
    """
    libraries = [steam_path]  # Main Steam folder is always a library
    
    vdf_path = Path(steam_path) / "steamapps" / "libraryfolders.vdf"
    if not vdf_path.exists():
        return libraries
    
    try:
        with open(vdf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple VDF parsing for "path" values
        # Format: "path" "C:\\SteamLibrary"
        import re
        path_pattern = r'"path"\s+"([^"]+)"'
        matches = re.findall(path_pattern, content)
        
        for match in matches:
            # Normalize path separators
            lib_path = match.replace("\\\\", "\\")
            if os.path.exists(lib_path):
                libraries.append(lib_path)
        
    except Exception:
        pass
    
    return libraries


def _parse_steam_manifest(manifest_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse a Steam appmanifest_*.acf file.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Dict with appid, name, installdir or None
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple ACF parsing (VDF format)
        import re
        
        appid_match = re.search(r'"appid"\s+"(\d+)"', content)
        name_match = re.search(r'"name"\s+"([^"]+)"', content)
        installdir_match = re.search(r'"installdir"\s+"([^"]+)"', content)
        
        if appid_match and name_match:
            return {
                "appid": appid_match.group(1),
                "name": name_match.group(1),
                "installdir": installdir_match.group(1) if installdir_match else ""
            }
        
        return None
        
    except Exception:
        return None


# ==============================================================================
# EPIC GAMES INDEXER
# ==============================================================================

def _index_epic_games() -> List[Dict[str, Any]]:
    """
    Index Epic Games from manifest files.
    
    Returns:
        List of game dicts
    """
    games = []
    
    manifests_path = Path(os.environ.get("PROGRAMDATA", "")) / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    
    if not manifests_path.exists():
        return games
    
    try:
        for manifest_file in manifests_path.glob("*.item"):
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                
                display_name = manifest_data.get("DisplayName", "")
                app_name = manifest_data.get("AppName", "")
                install_location = manifest_data.get("InstallLocation", "")
                
                if display_name and app_name:
                    # Build Epic Games launcher URI
                    launch_uri = f"com.epicgames.launcher://apps/{app_name}?action=launch&silent=true"
                    
                    games.append({
                        "name": display_name,
                        "source": "epic",
                        "launch": {
                            "type": "epic_uri",
                            "target": launch_uri
                        },
                        "install_path": install_location if install_location else None,
                        "app_id": app_name,
                        "aliases": _generate_game_aliases(display_name),
                        "confidence": 0.95
                    })
            except Exception:
                # Skip individual manifest errors
                pass
    except Exception:
        # Directory not accessible
        pass
    
    return games


# ==============================================================================
# SHORTCUT INDEXER
# ==============================================================================

def _index_game_shortcuts() -> List[Dict[str, Any]]:
    """
    Index game shortcuts from Start Menu and Desktop.
    
    Returns:
        List of game dicts
    """
    games = []
    
    # Define scan paths
    scan_paths = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    ]
    
    # Game-related keywords
    game_keywords = [
        "game", "play", "gaming", "steam", "epic", "gog", "origin", "uplay", "ubisoft",
        "riot", "blizzard", "battle.net", "battlenet", "minecraft", "roblox", "league",
        "rocket", "counter-strike", "cs", "valorant", "overwatch", "fortnite", "apex",
        "call of duty", "cod", "battlefield", "rust", "ark", "gta", "grand theft auto",
        "terraria", "stardew", "among us", "fall guys", "warzone", "destiny"
    ]
    
    for scan_path in scan_paths:
        if not scan_path.exists():
            continue
        
        try:
            for shortcut_file in _find_shortcuts(scan_path, max_depth=3):
                try:
                    shortcut_name = shortcut_file.stem
                    shortcut_name_lower = shortcut_name.lower()
                    
                    # Skip uninstaller shortcuts
                    if "uninstall" in shortcut_name_lower or "uninst" in shortcut_name_lower:
                        continue
                    
                    # Check if shortcut name suggests it's a game
                    is_game = any(keyword in shortcut_name_lower for keyword in game_keywords)
                    
                    # Also check target path for game indicators
                    if not is_game:
                        target_path = _get_shortcut_target(shortcut_file)
                        if target_path:
                            target_lower = target_path.lower()
                            is_game = any(keyword in target_lower for keyword in game_keywords)
                    
                    if is_game:
                        games.append({
                            "name": shortcut_name,
                            "source": "shortcut",
                            "launch": {
                                "type": "shortcut",
                                "target": str(shortcut_file)
                            },
                            "install_path": None,
                            "app_id": None,
                            "aliases": _generate_game_aliases(shortcut_name),
                            "confidence": 0.80
                        })
                except Exception:
                    # Skip individual shortcut errors
                    pass
        except Exception:
            # Skip scan path errors
            pass
    
    return games


def _find_shortcuts(path: Path, max_depth: int = 3, current_depth: int = 0) -> List[Path]:
    """
    Recursively find .lnk shortcuts in a directory.
    
    Args:
        path: Directory to scan
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        List of shortcut file paths
    """
    shortcuts = []
    
    if current_depth >= max_depth:
        return shortcuts
    
    try:
        for entry in path.iterdir():
            if entry.is_dir():
                shortcuts.extend(_find_shortcuts(entry, max_depth, current_depth + 1))
            elif entry.is_file() and entry.suffix.lower() == ".lnk":
                shortcuts.append(entry)
    except (PermissionError, OSError):
        pass
    
    return shortcuts


def _get_shortcut_target(shortcut_path: Path) -> Optional[str]:
    """
    Get target path from a .lnk shortcut file.
    
    Uses PowerShell as fallback if winshell not available.
    
    Args:
        shortcut_path: Path to .lnk file
        
    Returns:
        Target path string or None
    """
    # Try using winshell if available
    try:
        import winshell
        shortcut = winshell.shortcut(str(shortcut_path))
        return shortcut.path
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: Use PowerShell
    try:
        ps_command = f"""
        $ws = New-Object -ComObject WScript.Shell
        $shortcut = $ws.CreateShortcut('{shortcut_path}')
        $shortcut.TargetPath
        """
        
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode == 0:
            target = result.stdout.strip()
            return target if target else None
        
        return None
        
    except Exception:
        return None


# ==============================================================================
# FOLDER SCAN INDEXER
# ==============================================================================

def _index_folder_scan() -> List[Dict[str, Any]]:
    """
    Scan common game install folders for game executables.
    
    Returns:
        List of game dicts
    """
    games = []
    
    # Define scan locations
    scan_locations = [
        Path(r"C:\Games"),
        Path(r"D:\Games"),
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
    ]
    
    # Exclude directories
    exclude_dirs = {
        "windows", "system32", "winsxs", "installer", "drivers", "dotnet",
        "common files", "runtime", "temp", "cache", "updates", "$recycle.bin",
        "windowsapps", "microsoft.net", "windows defender"
    }
    
    # Game executable patterns
    game_patterns = [
        "game.exe", "shipping.exe", "launcher.exe", "play.exe",
        "unityplayer.dll", "unrealengine.dll", "steam_api.dll", "steam_api64.dll"
    ]
    
    # Game folder patterns
    game_folder_patterns = [
        "binaries/win64", "engine/binaries", "bin/win64", "bin64"
    ]
    
    for location in scan_locations:
        if not location.exists():
            continue
        
        try:
            games.extend(_scan_game_folder(
                location,
                exclude_dirs,
                game_patterns,
                game_folder_patterns,
                max_depth=3
            ))
        except Exception:
            # Skip location errors
            pass
    
    return games


def _scan_game_folder(
    path: Path,
    exclude_dirs: set,
    game_patterns: List[str],
    game_folder_patterns: List[str],
    max_depth: int = 3,
    current_depth: int = 0
) -> List[Dict[str, Any]]:
    """
    Recursively scan for game executables.
    
    Args:
        path: Directory to scan
        exclude_dirs: Set of directory names to exclude (lowercase)
        game_patterns: List of game file patterns
        game_folder_patterns: List of game folder patterns
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        List of game dicts
    """
    games = []
    
    if current_depth >= max_depth:
        return games
    
    try:
        # Check if current folder looks like a game folder
        folder_name_lower = path.name.lower()
        
        # Skip excluded directories
        if any(exclude in folder_name_lower for exclude in exclude_dirs):
            return games
        
        # Check for game indicators
        is_game_folder = False
        game_exe = None
        
        # Check for game patterns in current directory
        try:
            for entry in path.iterdir():
                if entry.is_file():
                    entry_name_lower = entry.name.lower()
                    
                    # Check for game patterns
                    if any(pattern in entry_name_lower for pattern in game_patterns):
                        is_game_folder = True
                        
                        # Prefer actual .exe files
                        if entry.suffix.lower() == ".exe" and "shipping" in entry_name_lower:
                            game_exe = entry
                            break
                        elif entry.suffix.lower() == ".exe" and game_exe is None:
                            game_exe = entry
        except Exception:
            pass
        
        # Check for game folder patterns
        path_str_lower = str(path).lower().replace("\\", "/")
        if any(pattern in path_str_lower for pattern in game_folder_patterns):
            is_game_folder = True
        
        # If this looks like a game folder and we found an exe
        if is_game_folder and game_exe:
            game_name = path.name if path.name not in ["Binaries", "Win64", "Bin"] else path.parent.name
            
            games.append({
                "name": game_name,
                "source": "folder_scan",
                "launch": {
                    "type": "exe",
                    "target": str(game_exe)
                },
                "install_path": str(path),
                "app_id": None,
                "aliases": _generate_game_aliases(game_name),
                "confidence": 0.70
            })
        
        # Recurse into subdirectories
        try:
            for entry in path.iterdir():
                if entry.is_dir():
                    games.extend(_scan_game_folder(
                        entry,
                        exclude_dirs,
                        game_patterns,
                        game_folder_patterns,
                        max_depth,
                        current_depth + 1
                    ))
        except Exception:
            pass
        
    except (PermissionError, OSError):
        pass
    
    return games


# ==============================================================================
# XBOX / MICROSOFT STORE INDEXER
# ==============================================================================

def _index_xbox_games() -> List[Dict[str, Any]]:
    """
    Index Xbox/Microsoft Store games via PowerShell Get-AppxPackage.
    
    Returns:
        List of game dicts
    """
    games = []
    
    try:
        # Get AppX packages via PowerShell
        ps_command = "Get-AppxPackage | Select-Object Name, PackageFamilyName | ConvertTo-Json"
        
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode != 0:
            return games
        
        # Parse JSON output
        packages_data = json.loads(result.stdout)
        
        # Handle single item (not array)
        if isinstance(packages_data, dict):
            packages_data = [packages_data]
        
        # Filter for game-related packages
        game_keywords = [
            "game", "xbox", "minecraft", "solitaire", "candy", "farm", "mahjong",
            "casino", "puzzle", "racing", "sports", "action", "adventure", "shooter"
        ]
        
        for package in packages_data:
            name = package.get("Name", "")
            family_name = package.get("PackageFamilyName", "")
            
            name_lower = name.lower()
            
            # Check if package name suggests it's a game
            if any(keyword in name_lower for keyword in game_keywords):
                # Clean up name
                display_name = name.split(".")[-1] if "." in name else name
                display_name = display_name.replace("_", " ").title()
                
                games.append({
                    "name": display_name,
                    "source": "xbox",
                    "launch": {
                        "type": "uwp",
                        "target": f"shell:AppsFolder\\{family_name}!App"
                    },
                    "install_path": None,
                    "app_id": family_name,
                    "aliases": _generate_game_aliases(display_name),
                    "confidence": 0.75
                })
        
    except subprocess.TimeoutExpired:
        # PowerShell took too long
        pass
    except json.JSONDecodeError:
        # Failed to parse output
        pass
    except Exception:
        # Other errors (permissions, etc.)
        pass
    
    return games


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _generate_game_aliases(game_name: str) -> List[str]:
    """
    Generate common aliases for a game name.
    
    Args:
        game_name: Original game name
        
    Returns:
        List of alias strings (lowercase)
    """
    aliases = [game_name.lower()]
    
    # Remove common suffixes
    clean_name = game_name
    suffixes = [" - ", ": ", "(", " Game", " Launcher"]
    for suffix in suffixes:
        if suffix in clean_name:
            clean_name = clean_name.split(suffix)[0]
    
    if clean_name.lower() != game_name.lower():
        aliases.append(clean_name.lower())
    
    # Common abbreviations
    abbreviations = {
        "rocket league": ["rl"],
        "counter-strike": ["cs", "csgo"],
        "call of duty": ["cod"],
        "grand theft auto": ["gta"],
        "league of legends": ["lol"],
        "world of warcraft": ["wow"],
        "overwatch": ["ow"],
        "rainbow six": ["r6"],
        "battlefield": ["bf"],
        "apex legends": ["apex"],
        "destiny": ["d2"],
        "the witcher": ["witcher"]
    }
    
    name_lower = game_name.lower()
    for full_name, abbrevs in abbreviations.items():
        if full_name in name_lower:
            aliases.extend(abbrevs)
    
    # Remove duplicates
    return list(set(aliases))


def _deduplicate_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate games by name, preferring higher confidence sources.
    
    Args:
        games: List of game dicts
        
    Returns:
        Deduplicated list
    """
    # Group by normalized name
    games_by_name = {}
    
    for game in games:
        name_key = game["name"].lower().strip()
        
        if name_key not in games_by_name:
            games_by_name[name_key] = game
        else:
            # Keep the higher confidence one
            existing_game = games_by_name[name_key]
            if game["confidence"] > existing_game["confidence"]:
                games_by_name[name_key] = game
            # If same confidence, prefer certain sources
            elif game["confidence"] == existing_game["confidence"]:
                source_priority = {"steam": 5, "epic": 4, "shortcut": 3, "folder_scan": 2, "xbox": 1}
                if source_priority.get(game["source"], 0) > source_priority.get(existing_game["source"], 0):
                    games_by_name[name_key] = game
    
    return list(games_by_name.values())
