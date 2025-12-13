"""
Base class for all Wyzer tools.
Tools are stateless functions that return JSON-serializable dicts.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolBase(ABC):
    """Base class for all tools"""
    
    def __init__(self):
        """Initialize tool with metadata"""
        self._name: str = ""
        self._description: str = ""
        self._args_schema: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Tool name (unique identifier)"""
        return self._name
    
    @property
    def description(self) -> str:
        """Human-readable description of what the tool does"""
        return self._description
    
    @property
    def args_schema(self) -> Dict[str, Any]:
        """JSON-schema-like definition of expected arguments"""
        return self._args_schema
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with provided arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            JSON-serializable dict with results or error
        """
        pass
