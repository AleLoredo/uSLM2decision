"""
OpenJEV Dynamic Prometheus Metrics Collector
Implements IMPLEMENTATION.md §8 metrics breakdown:
- Latency histogram labeled by primitive (bool, choice, score)
- GPU evaluation duration separated from network/serialization overhead
- Live queue depth gauge and request counters
"""

import threading
from typing import Dict, List, Tuple
from collections import defaultdict

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)

class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        # Key: (primitive, status_code) -> count
        self.request_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # Key: primitive -> {bucket_le: count}
        self.latency_buckets: Dict[str, Dict[float, int]] = defaultdict(lambda: {b: 0 for b in LATENCY_BUCKETS})
        self.latency_sums: Dict[str, float] = defaultdict(float)
        self.latency_counts: Dict[str, int] = defaultdict(int)

        # GPU Evaluation Duration
        self.gpu_buckets: Dict[float, int] = {b: 0 for b in LATENCY_BUCKETS}
        self.gpu_sum: float = 0.0
        self.gpu_count: int = 0

        # Queue Depth
        self.queue_depth: int = 0

    def record_request(
        self,
        primitive: str,
        status_code: int,
        latency_sec: float,
        gpu_eval_sec: float = 0.0
    ):
        with self._lock:
            # 1. Total Requests Counter
            self.request_counts[(primitive, status_code)] += 1

            # 2. Total Latency Histogram per primitive
            p_buckets = self.latency_buckets[primitive]
            for b in LATENCY_BUCKETS:
                if latency_sec <= b:
                    p_buckets[b] += 1
            self.latency_sums[primitive] += latency_sec
            self.latency_counts[primitive] += 1

            # 3. GPU Evaluation Duration Histogram
            if gpu_eval_sec > 0.0:
                for b in LATENCY_BUCKETS:
                    if gpu_eval_sec <= b:
                        self.gpu_buckets[b] += 1
                self.gpu_sum += gpu_eval_sec
                self.gpu_count += 1

    def set_queue_depth(self, depth: int):
        with self._lock:
            self.queue_depth = max(0, depth)

    def generate_prometheus_text(self) -> str:
        lines: List[str] = [
            "# HELP openjev_requests_total Total decision requests processed",
            "# TYPE openjev_requests_total counter"
        ]

        with self._lock:
            if not self.request_counts:
                lines.append('openjev_requests_total{primitive="none",status="200"} 0')
            else:
                for (prim, st), count in sorted(self.request_counts.items()):
                    lines.append(f'openjev_requests_total{{primitive="{prim}",status="{st}"}} {count}')

            # Latency histograms per primitive
            lines.extend([
                "# HELP openjev_latency_seconds Request end-to-end latency in seconds",
                "# TYPE openjev_latency_seconds histogram"
            ])
            for prim in sorted(self.latency_buckets.keys()):
                p_buckets = self.latency_buckets[prim]
                cum = 0
                for b in LATENCY_BUCKETS:
                    cum = p_buckets[b]
                    lines.append(f'openjev_latency_seconds_bucket{{primitive="{prim}",le="{b}"}} {cum}')
                lines.append(f'openjev_latency_seconds_bucket{{primitive="{prim}",le="+Inf"}} {self.latency_counts[prim]}')
                lines.append(f'openjev_latency_seconds_sum{{primitive="{prim}"}} {self.latency_sums[prim]:.6f}')
                lines.append(f'openjev_latency_seconds_count{{primitive="{prim}"}} {self.latency_counts[prim]}')

            # GPU eval histogram
            lines.extend([
                "# HELP openjev_gpu_eval_duration_seconds Pure Metal GPU forward-pass compute duration in seconds",
                "# TYPE openjev_gpu_eval_duration_seconds histogram"
            ])
            for b in LATENCY_BUCKETS:
                lines.append(f'openjev_gpu_eval_duration_seconds_bucket{{le="{b}"}} {self.gpu_buckets[b]}')
            lines.append(f'openjev_gpu_eval_duration_seconds_bucket{{le="+Inf"}} {self.gpu_count}')
            lines.append(f'openjev_gpu_eval_duration_seconds_sum {self.gpu_sum:.6f}')
            lines.append(f'openjev_gpu_eval_duration_seconds_count {self.gpu_count}')

            # Queue depth gauge
            lines.extend([
                "# HELP openjev_queue_depth Current pending queue depth in inference engine",
                "# TYPE openjev_queue_depth gauge",
                f"openjev_queue_depth {self.queue_depth}"
            ])

        return "\n".join(lines) + "\n"

# Global singleton collector
metrics_collector = MetricsCollector()
