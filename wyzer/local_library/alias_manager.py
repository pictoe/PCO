"""Utilities for managing LocalLibrary aliases.

Aliases are stored in wyzer/local_library/aliases.json, and merged into the in-memory
library index for resolution.

This module is intentionally small and dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ALIASES_PATH = Path(__file__).parent / "aliases.json"


@dataclass(frozen=True)
class AliasWriteResult:
    status: str  # created|exists|conflict|invalid
    message: str
    alias: str
    target_type: Optional[str] = None
    target: Optional[str] = None


def _load_aliases_raw() -> Dict[str, Any]:
    if not ALIASES_PATH.exists():
        return {}
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_aliases_raw(data: Dict[str, Any]) -> None:
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALIASES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_alias(
    alias: str,
    target_type: str,
    target: str,
    *,
    overwrite: bool = False,
) -> AliasWriteResult:
    """Add/update an alias entry in aliases.json.

    - Normalizes alias to lowercase.
    - Does not overwrite existing entries unless overwrite=True.
    """
    alias_norm = (alias or "").strip().lower()
    if not alias_norm:
        return AliasWriteResult(status="invalid", message="Alias cannot be empty", alias="")
    if not target_type or not target:
        return AliasWriteResult(status="invalid", message="Target type and target are required", alias=alias_norm)

    data = _load_aliases_raw()

    existing = data.get(alias_norm)
    new_value = {"type": target_type, "target": target}

    if isinstance(existing, dict) and existing.get("type") == target_type and existing.get("target") == target:
        return AliasWriteResult(
            status="exists",
            message="Alias already exists",
            alias=alias_norm,
            target_type=target_type,
            target=target,
        )

    if existing is not None and not overwrite:
        return AliasWriteResult(
            status="conflict",
            message="Alias exists with different target (set overwrite=True to replace)",
            alias=alias_norm,
            target_type=target_type,
            target=target,
        )

    data[alias_norm] = new_value
    _save_aliases_raw(data)

    return AliasWriteResult(
        status="created",
        message="Alias saved",
        alias=alias_norm,
        target_type=target_type,
        target=target,
    )


def has_alias(alias: str) -> bool:
    alias_norm = (alias or "").strip().lower()
    if not alias_norm:
        return False
    data = _load_aliases_raw()
    return alias_norm in data
