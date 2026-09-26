"""
OpenJEV Engine Factory
Instantiates the appropriate decision engine based on OpenJEVConfig.
"""

from core.config import OpenJEVConfig
from core.engine import BaseInferenceEngine
from core.engine_mock import MockInferenceEngine

def create_engine(config: OpenJEVConfig) -> BaseInferenceEngine:
    """Factory function returning the configured inference engine."""
    if config.mode == "mock":
        return MockInferenceEngine(config)
    elif config.mode == "real":
        if config.model.backend == "mlx":
            from core.engine_mlx import MLXInferenceEngine
            return MLXInferenceEngine(config)
        else:
            raise NotImplementedError(
                f"Backend '{config.model.backend}' is not yet implemented for real mode."
            )
    else:
        raise ValueError(f"Unknown OpenJEV mode: '{config.mode}'")
