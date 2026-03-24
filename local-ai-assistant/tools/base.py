"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import Callable


class BaseTool(ABC):
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def get_tool_definitions(self) -> list[dict]:
        """Return Ollama-format tool definitions."""
        ...
    
    @abstractmethod
    def get_handlers(self) -> dict[str, Callable]:
        """Return map of function_name → callable."""
        ...
