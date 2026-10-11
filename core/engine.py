"""
OpenJEV Abstract Engine Interface
Defines the normative contract for System 1 decision inference engines per SPEC.md and ARCHITECTURE.md.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

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
    def decide(self, state: Union[str, Dict[str, Any], List[Any]], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes decision questions over state in a single forward pass.
        Returns a dictionary containing 'answers' and 'usage'.
        """
        pass

    def decide_batch(self, requests: List[Any]) -> List[Dict[str, Any]]:
        """
        Executes a batch of requests in a forward pass.
        Default implementation evaluates each request sequentially.
        """
        results = []
        for req in requests:
            st = getattr(req, "state", None) if hasattr(req, "state") else req.get("state", "")
            qs = getattr(req, "questions", None) if hasattr(req, "questions") else req.get("questions", {})
            results.append(self.decide(st, qs))
        return results

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifier of active model version."""
        pass

    @property
    @abstractmethod
    def calibration_version(self) -> str:
        """Identifier of active calibration artifact version."""
        pass
