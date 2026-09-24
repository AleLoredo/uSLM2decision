"""
OpenDecide Self-Hosted Recalibration Recipe
Per ROADMAP.md Phase 4:
"Bring ~200 labeled examples from your domain, refit temperature in under a minute."
"""

import time
from typing import Dict, List, Any, Tuple
import mlx.core as mx
from calibration.post_hoc import TemperatureScaler

def recalibrate_domain_model(
    domain_logits: mx.array,
    domain_targets: mx.array,
    primitive_type: str = "bool"
) -> Dict[str, Any]:
    """
    Fast self-hosted recalibration routine. Refits temperature scaling parameters
    on domain-specific validation examples in seconds.
    
    Args:
        domain_logits: Logits from 200 domain-specific examples
        domain_targets: Ground truth domain targets
        primitive_type: 'bool', 'choice', or 'score'
    
    Returns:
        Dict containing fitted temperature, execution time in seconds, and status
    """
    start_time = time.perf_counter()
    
    scaler = TemperatureScaler(initial_temperature=1.0)
    fitted_temp = scaler.fit(domain_logits, domain_targets, search_range=(0.2, 5.0), steps=100)
    
    elapsed_sec = time.perf_counter() - start_time
    
    return {
        "status": "success",
        "primitive_type": primitive_type,
        "fitted_temperature": fitted_temp,
        "num_examples": domain_logits.shape[0],
        "elapsed_seconds": elapsed_sec,
        "message": f"Recalibration completed in {elapsed_sec * 1000.0:.2f} ms on {domain_logits.shape[0]} domain examples."
    }
