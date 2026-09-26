"""
OpenJEV Micro-Batching Thread-Safe Queue
Implements bounded accumulation with deadline-based dynamic batching per IMPLEMENTATION.md §4.
"""

import time
import queue
from typing import List

from core.batching.types import QueuedRequest, QueueFullError

class BatchQueue:
    def __init__(self, max_batch_size: int = 16, max_wait_ms: int = 8, max_queue_depth: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_queue_depth = max_queue_depth
        self._queue: queue.Queue[QueuedRequest] = queue.Queue(maxsize=max_queue_depth)
        self._closed = False

    def qsize(self) -> int:
        return self._queue.qsize()

    def put(self, req: QueuedRequest) -> None:
        """Enqueues a request or raises QueueFullError if queue depth is exceeded."""
        if self._closed:
            raise RuntimeError("BatchQueue is closed")
        try:
            self._queue.put_nowait(req)
        except queue.Full:
            raise QueueFullError(
                f"Queue depth limit ({self.max_queue_depth}) reached. Backpressure triggered."
            )

    def get_batch(self, timeout: float = 0.05) -> List[QueuedRequest]:
        """
        Pulls a micro-batch of requests.
        Blocks for the first item up to `timeout`, then waits up to max_wait_ms
        to accumulate up to max_batch_size requests.
        """
        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            return []

        batch = [first]
        deadline = time.time() + (self.max_wait_ms / 1000.0)

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

        return batch

    def close(self) -> None:
        """Closes the queue."""
        self._closed = True
