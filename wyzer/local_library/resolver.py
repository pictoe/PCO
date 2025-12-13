"""
Resolver for LocalLibrary - matches user queries to indexed targets.
"""
from typing import Dict, Any, List
from pathlib import Path
from wyzer.local_library.indexer import get_cached_index


def resolve_target(query: str) -> Dict[str, Any]:
    """
    Resolve a user query to a target (folder, file, app, game, url).
    
    Args:
        query: User query like "downloads", "my minecraft folder", "chrome", "play rocket league"
        
    Returns:
        {
            "type": "folder|file|app|game|url|unknown",
            "path": "...",  # or "url" for url type, or launch info for games
            "confidence": float (0-1),
            "candidates": [...]  # alternative matches
        }
    """
    # Normalize query
    query_lower = query.strip().lower()
    
    # Check if query suggests game intent
    game_intent_keywords = ["play", "launch", "start game", "open game"]
    has_game_intent = any(keyword in query_lower for keyword in game_intent_keywords)
    
    # Get cached index
    index = get_cached_index()
    
    # Try exact matches first
    result = _try_exact_match(query_lower, index, has_game_intent)
    if result:
        return result
    
    # Try fuzzy matching
    result = _try_fuzzy_match(query_lower, index, has_game_intent)
    if result:
        return result
    
    # Check if it looks like a URL
    if _looks_like_url(query):
        return {
            "type": "url",
            "url": query,
            "confidence": 0.9,
            "candidates": []
        }
    
    # No match found
    return {
        "type": "unknown",
        "path": "",
        "confidence": 0.0,
        "candidates": []
    }


def _try_exact_match(query: str, index: Dict[str, Any], has_game_intent: bool = False) -> Dict[str, Any]:
    """
    Try to find an exact match in the index.
    
    Args:
        query: Normalized query
        index: Library index
        has_game_intent: Whether query suggests game intent
    
    Returns:
        Result dict or None if no match
    """
    # Check aliases first (highest priority)
    aliases = index.get("aliases", {})
    if query in aliases:
        alias_data = aliases[query]
        return {
            "type": alias_data.get("type", "unknown"),
            "path": alias_data.get("target", ""),
            "confidence": 1.0,
            "candidates": []
        }
    
    # Check games (high priority if game intent)
    games = index.get("games", [])
    for game in games:
        # Normalize game name for comparison
        game_name_normalized = _normalize_game_name(game["name"])
        
        # Check exact name match
        if query == game_name_normalized:
            return {
                "type": "game",
                "path": game["launch"]["target"],
                "launch": game["launch"],
                "game_name": game["name"],
                "confidence": game["confidence"],
                "candidates": []
            }
        
        # Check aliases
        if query in game.get("aliases", []):
            return {
                "type": "game",
                "path": game["launch"]["target"],
                "launch": game["launch"],
                "game_name": game["name"],
                "confidence": game["confidence"] * 0.95,  # Slightly lower for alias match
                "candidates": []
            }
    
    # If game intent is strong, don't check non-game targets
    if has_game_intent:
        return None
    
    # Check UWP apps
    uwp_apps = index.get("uwp_apps", [])
    for uwp_app in uwp_apps:
        # Check exact name match
        if query == uwp_app["name"].lower():
            return {
                "type": "uwp",
                "path": uwp_app["app_id"],
                "confidence": 0.95,
                "launch": {"type": "uwp", "target": uwp_app["app_id"]},
                "app_name": uwp_app["name"],
                "candidates": []
            }
        
        # Check aliases
        if query in uwp_app.get("aliases", []):
            return {
                "type": "uwp",
                "path": uwp_app["app_id"],
                "confidence": 0.90,
                "launch": {"type": "uwp", "target": uwp_app["app_id"]},
                "app_name": uwp_app["name"],
                "candidates": []
            }
    
    # Check folders
    folders = index.get("folders", {})
    if query in folders:
        return {
            "type": "folder",
            "path": folders[query],
            "confidence": 1.0,
            "candidates": []
        }
    
    # Check apps (Start Menu - higher priority)
    apps = index.get("apps", {})
    if query in apps:
        app_data = apps[query]
        return {
            "type": "app",
            "path": app_data.get("path", ""),
            "confidence": 0.95,  # Start Menu apps have high confidence
            "candidates": []
        }
    
    # Check tier2_apps (lower priority)
    tier2_apps = index.get("tier2_apps", [])
    for app in tier2_apps:
        app_name_lower = app["name"].lower()
        if query == app_name_lower:
            return {
                "type": "app",
                "path": app["exe_path"],
                "confidence": 0.90,  # Tier 2 exact match
                "candidates": []
            }
    
    return None


def _try_fuzzy_match(query: str, index: Dict[str, Any], has_game_intent: bool = False) -> Dict[str, Any]:
    """
    Try to find a fuzzy match in the index.
    
    Uses simple substring and word matching.
    Merges Start Menu apps with Tier 2 apps, preferring Start Menu.
    Prioritizes games when game intent is detected.
    
    Args:
        query: Normalized query
        index: Library index
        has_game_intent: Whether query suggests game intent
    
    Returns:
        Result dict or None if no match
    """
    candidates = []
    
    # Extract keywords from query
    keywords = _extract_keywords(query)
    
    # Search games (high priority if game intent)
    games = index.get("games", [])
    for game in games:
        game_name_normalized = _normalize_game_name(game["name"])
        score = _match_score(keywords, game_name_normalized)
        
        # Also check aliases
        for alias in game.get("aliases", []):
            alias_score = _match_score(keywords, alias)
            score = max(score, alias_score)
        
        if score > 0.3:
            # Boost confidence if game intent detected
            confidence = game["confidence"] * score
            if has_game_intent:
                confidence = min(confidence * 1.15, 0.98)
            
            candidates.append({
                "type": "game",
                "path": game["launch"]["target"],
                "launch": game["launch"],
                "game_name": game["name"],
                "confidence": confidence,
                "name": game["name"],
                "source": "game"
            })
    
    # If no game intent, search other targets
    if not has_game_intent:
        # Search UWP apps
        uwp_apps = index.get("uwp_apps", [])
        for uwp_app in uwp_apps:
            app_name_lower = uwp_app["name"].lower()
            score = _match_score(keywords, app_name_lower)
            
            # Also check aliases
            for alias in uwp_app.get("aliases", []):
                alias_score = _match_score(keywords, alias)
                score = max(score, alias_score)
            
            if score > 0.3:
                # Base confidence for UWP
                confidence = score * 0.85
                
                # Boost for exact keyword match
                if any(kw == app_name_lower for kw in keywords):
                    confidence = min(confidence * 1.1, 0.95)
                
                candidates.append({
                    "type": "uwp",
                    "path": uwp_app["app_id"],
                    "confidence": confidence,
                    "launch": {"type": "uwp", "target": uwp_app["app_id"]},
                    "app_name": uwp_app["name"],
                    "name": uwp_app["name"],
                    "source": "uwp"
                })
        
        # Search folders
        folders = index.get("folders", {})
        for folder_name, folder_path in folders.items():
            score = _match_score(keywords, folder_name)
            if score > 0.3:
                candidates.append({
                    "type": "folder",
                    "path": folder_path,
                    "confidence": score,
                    "name": folder_name,
                    "source": "folder"
                })
        
        # Search Start Menu apps (higher priority)
        apps = index.get("apps", {})
        for app_name, app_data in apps.items():
            score = _match_score(keywords, app_name)
            if score > 0.3:
                # Boost confidence for Start Menu apps
                confidence = min(score * 1.05, 0.95)
                candidates.append({
                    "type": "app",
                    "path": app_data.get("path", ""),
                    "confidence": confidence,
                    "name": app_name,
                    "source": "start_menu"
                })
        
        # Search Tier 2 apps
        tier2_apps = index.get("tier2_apps", [])
        for app in tier2_apps:
            app_name_lower = app["name"].lower()
            score = _match_score(keywords, app_name_lower)
            
            if score > 0.3:
                # Base confidence for Tier 2
                confidence = score * 0.85
                
                # Boost confidence for exact keyword matches
                if any(kw == app_name_lower for kw in keywords):
                    confidence = min(confidence * 1.1, 0.90)
                
                # Prefer shorter paths (likely more direct installs)
                path_depth = app["exe_path"].count("\\")
                if path_depth <= 4:
                    confidence = min(confidence * 1.05, 0.90)
                
                candidates.append({
                    "type": "app",
                    "path": app["exe_path"],
                    "confidence": confidence,
                    "name": app["name"],
                    "source": "tier2"
                })
    
    # Deduplicate: prefer Start Menu over Tier 2 for same app
    seen_names = {}
    unique_candidates = []
    
    # Sort by confidence first
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    for candidate in candidates:
        name_key = candidate["name"].lower()
        
        # If we haven't seen this name, add it
        if name_key not in seen_names:
            seen_names[name_key] = True
            unique_candidates.append(candidate)
        # If we've seen it, only add if it's from a higher priority source
        elif candidate["source"] == "start_menu" and candidate["confidence"] > 0.8:
            # Replace if this Start Menu entry is better
            unique_candidates.append(candidate)
    
    # Re-sort after deduplication
    unique_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    if unique_candidates:
        best = unique_candidates[0]
        result = {
            "type": best["type"],
            "path": best["path"],
            "confidence": best["confidence"],
            "candidates": unique_candidates[1:5]  # Include up to 4 alternatives
        }
        
        # Include launch info for games
        if best["type"] == "game":
            result["launch"] = best.get("launch", {})
            result["game_name"] = best.get("game_name", "")
        
        # Include launch info for UWP apps
        if best["type"] == "uwp":
            result["launch"] = best.get("launch", {})
            result["app_name"] = best.get("app_name", "")
        
        return result
    
    return None


def _extract_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from query.
    
    Args:
        query: User query string
        
    Returns:
        List of keywords
    """
    # Remove common filler words
    stop_words = {"my", "the", "a", "an", "open", "launch", "start", "run", "show", "go", "to", "folder", "app", "application"}
    
    words = query.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 1]
    
    return keywords


def _match_score(keywords: List[str], target: str) -> float:
    """
    Calculate match score between keywords and target.
    
    Args:
        keywords: List of query keywords
        target: Target name to match against
        
    Returns:
        Score between 0 and 1
    """
    if not keywords:
        return 0.0
    
    matches = 0
    for keyword in keywords:
        if keyword in target:
            matches += 1
        elif any(keyword in part for part in target.split()):
            matches += 0.5
    
    return matches / len(keywords)


def _looks_like_url(query: str) -> bool:
    """
    Check if query looks like a URL.
    
    Args:
        query: User query
        
    Returns:
        True if looks like URL
    """
    query_lower = query.lower()
    
    # Check for URL schemes
    if query_lower.startswith(("http://", "https://", "www.")):
        return True
    
    # Check for domain-like patterns
    if "." in query and " " not in query:
        # Simple heuristic: contains dot, no spaces
        parts = query.split(".")
        if len(parts) >= 2 and all(part.isalnum() for part in parts):
            return True
    
    return False


def _normalize_game_name(name: str) -> str:
    """
    Normalize game name for matching by removing special characters and symbols.
    
    Args:
        name: Game name
        
    Returns:
        Normalized name (lowercase, no special chars)
    """
    import re
    # Remove trademark symbols and other special characters
    normalized = re.sub(r'[®™©]', '', name)
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    return normalized.lower()
