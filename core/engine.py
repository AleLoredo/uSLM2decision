"""
OpenJEV Abstract Engine Interface
Defines the normative contract for decision inference engines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseInferenceEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Loads backbone, tokenizer, and decision heads into memory/GPU."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the engine is ready to serve requests."""
        pass

    @abstractmethod
    def decide(self, context: str, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes decision queries over context in a single forward pass."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifier of active model version."""
        pass

    @property
    @abstractmethod
    def calibration_version(self) -> str:
        """Identifier of active calibration version."""
        pass
