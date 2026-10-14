"""
OpenJEV Batching Types
Defines the data structures for dynamic request micro-batching conforming to ARCHITECTURE.md.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class QueuedRequest:
    request_id: str
    state: Union[str, Dict[str, Any], list]
    questions: Dict[str, Dict[str, Any]]
    model: Optional[str] = "default"
    arrival_time: float = field(default_factory=time.time)
    loop: Optional[asyncio.AbstractEventLoop] = None
    future: Optional[asyncio.Future] = None

class QueueFullError(Exception):
    """Raised when the micro-batch queue exceeds max_queue_depth."""
    pass
