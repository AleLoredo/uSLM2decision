"""
OpenJEV Batch Dispatcher
Dedicated MLX/Metal inference thread for serialized command buffer submission
per IMPLEMENTATION.md §4.
"""

import time
import threading
import logging
from typing import Optional

from core.engine import BaseInferenceEngine
from core.batching.queue import BatchQueue

logger = logging.getLogger("openjev.dispatcher")

class BatchDispatcher:
    def __init__(self, engine: BaseInferenceEngine, queue: BatchQueue):
        self.engine = engine
        self.queue = queue
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the dedicated dispatcher background thread if not already alive."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stopping = False
                self._thread = threading.Thread(
                    target=self._worker_loop,
                    name="mlx_dispatcher_thread",
                    daemon=True
                )
                self._thread.start()
                logger.info("[BatchDispatcher] Dedicated MLX Metal GPU thread started.")

    def stop(self, timeout: float = 2.0) -> None:
        """Stops the dispatcher worker thread cleanly."""
        self._stopping = True
        self.queue.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("[BatchDispatcher] Dedicated MLX Metal GPU thread stopped.")

    def _worker_loop(self) -> None:
        """Continuously pulls micro-batches and executes forward pass on Metal GPU."""
        while not self._stopping:
            try:
                batch = self.queue.get_batch(timeout=0.05)
                if not batch:
                    continue

                t0 = time.perf_counter()
                try:
                    # Forward pass through engine
                    results = self.engine.decide_batch(batch)
                    t_eval = time.perf_counter() - t0

                    try:
                        from server.metrics import RECORD_GPU_EVAL
                        RECORD_GPU_EVAL(t_eval)
                    except Exception:
                        pass

                    # Resolve futures on their respective event loops
                    for req, res in zip(batch, results):
                        if req.future and req.loop and not req.future.done():
                            req.loop.call_soon_threadsafe(req.future.set_result, res)

                except Exception as exc:
                    logger.error(f"[BatchDispatcher] Batch inference error: {exc}", exc_info=True)
                    for req in batch:
                        if req.future and req.loop and not req.future.done():
                            req.loop.call_soon_threadsafe(req.future.set_exception, exc)

            except Exception as e:
                logger.error(f"[BatchDispatcher] Unexpected error in worker thread: {e}", exc_info=True)
                time.sleep(0.01)
