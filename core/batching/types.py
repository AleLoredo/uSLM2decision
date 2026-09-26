"""
OpenJEV Batching Types
Defines the data structures for dynamic request micro-batching.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class QueuedRequest:
    request_id: str
    context: str
    queries: List[Dict[str, Any]]
    arrival_time: float = field(default_factory=time.time)
    loop: Optional[asyncio.AbstractEventLoop] = None
    future: Optional[asyncio.Future] = None

class QueueFullError(Exception):
    """Raised when the micro-batch queue exceeds max_queue_depth."""
    pass
