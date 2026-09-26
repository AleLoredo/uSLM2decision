"""
Unit tests for Live Dynamic Prometheus Metrics (IMPLEMENTATION.md §8)
"""

import pytest
from server.metrics import MetricsCollector

def test_metrics_collector_recording_and_prometheus_output():
    collector = MetricsCollector()
    collector.record_request(primitive="bool", status_code=200, latency_sec=0.042, gpu_eval_sec=0.038)
    collector.record_request(primitive="choice", status_code=200, latency_sec=0.085, gpu_eval_sec=0.079)
    collector.record_request(primitive="score", status_code=400, latency_sec=0.005, gpu_eval_sec=0.0)
    collector.set_queue_depth(3)

    text = collector.generate_prometheus_text()

    # Verify counters
    assert 'openjev_requests_total{primitive="bool",status="200"} 1' in text
    assert 'openjev_requests_total{primitive="choice",status="200"} 1' in text
    assert 'openjev_requests_total{primitive="score",status="400"} 1' in text

    # Verify histograms
    assert 'openjev_latency_seconds_bucket{primitive="bool",le="0.05"} 1' in text
    assert 'openjev_gpu_eval_duration_seconds_count 2' in text

    # Verify queue depth gauge
    assert "openjev_queue_depth 3" in text
