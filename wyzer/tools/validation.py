"""
Argument validation for tools.
Simple JSON-schema-like validation without external dependencies.
"""
from typing import Dict, Any, Tuple, Optional


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate arguments against a JSON-schema-like schema.
    
    Args:
        schema: Schema definition with type, properties, required, etc.
        args: Arguments to validate
        
    Returns:
        Tuple of (is_valid, error_dict_or_none)
    """
    # Check for required fields
    required = schema.get("required", [])
    for field in required:
        if field not in args:
            return False, {
                "error": "missing_required_field",
                "field": field,
                "message": f"Required field '{field}' is missing"
            }
    
    # Check for unknown fields (unless additionalProperties is True)
    properties = schema.get("properties", {})
    additional_allowed = schema.get("additionalProperties", False)
    
    if not additional_allowed:
        for key in args:
            if key not in properties:
                return False, {
                    "error": "unknown_field",
                    "field": key,
                    "message": f"Unknown field '{key}' is not allowed"
                }
    
    # Validate each property
    for key, value in args.items():
        if key in properties:
            prop_schema = properties[key]
            is_valid, error = _validate_value(key, value, prop_schema)
            if not is_valid:
                return False, error
    
    return True, None


def _validate_value(key: str, value: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate a single value against its schema"""
    expected_type = schema.get("type")
    
    if expected_type is None:
        return True, None  # No type constraint
    
    # Type checking
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list
    }
    
    python_type = type_map.get(expected_type)
    if python_type is None:
        return True, None  # Unknown type, skip validation
    
    if not isinstance(value, python_type):
        return False, {
            "error": "type_mismatch",
            "field": key,
            "expected_type": expected_type,
            "actual_type": type(value).__name__,
            "message": f"Field '{key}' expects type '{expected_type}' but got '{type(value).__name__}'"
        }
    
    # Additional constraints (optional, minimal support)
    if expected_type == "string":
        min_length = schema.get("minLength")
        if min_length and len(value) < min_length:
            return False, {
                "error": "string_too_short",
                "field": key,
                "min_length": min_length,
                "message": f"Field '{key}' must be at least {min_length} characters"
            }
        
        pattern = schema.get("pattern")
        if pattern:
            import re
            if not re.match(pattern, value):
                return False, {
                    "error": "pattern_mismatch",
                    "field": key,
                    "pattern": pattern,
                    "message": f"Field '{key}' does not match required pattern"
                }
    
    return True, None
