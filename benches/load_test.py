"""
OpenJEV Multi-Client Concurrency and SLA Benchmark Script
Evaluates p50, p90, p99 latencies, throughput (req/sec), and queue stability
across concurrency levels [1, 4, 16, 64] and sequence lengths [256, 512, 1024].
"""

import time
import asyncio
import numpy as np
import httpx

SERVER_URL = "http://127.0.0.1:8000/v1/decide"

def build_payload(context_len: int) -> dict:
    words = ["system", "policy", "security", "token", "access", "admin", "rule", "audit"]
    context = " ".join(words[i % len(words)] for i in range(context_len))
    return {
        "context": context,
        "queries": [
            {
                "id": "q1",
                "type": "bool",
                "criteria": "Is administrator access verified?"
            },
            {
                "id": "q2",
                "type": "choice",
                "criteria": "Authorization tier",
                "options": [
                    {"id": "tier1", "description": "High privilege"},
                    {"id": "tier2", "description": "Standard privilege"}
                ]
            },
            {
                "id": "q3",
                "type": "score",
                "min": 1,
                "max": 5,
                "rubric": "Security score 1-5"
            }
        ]
    }

async def benchmark_client(
    client: httpx.AsyncClient,
    payload: dict,
    total_requests: int,
    concurrency: int
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    status_codes = []

    async def send_one():
        async with sem:
            t0 = time.perf_counter()
            try:
                res = await client.post(SERVER_URL, json=payload, timeout=30.0)
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)
                status_codes.append(res.status_code)
            except Exception as e:
                latencies.append(9999.0)
                status_codes.append(500)

    start_wall = time.perf_counter()
    tasks = [asyncio.create_task(send_one()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start_wall

    arr = np.array(latencies)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    rps = float(total_requests / total_duration) if total_duration > 0 else 0.0
    success_rate = float(sum(1 for c in status_codes if c == 200) / total_requests * 100.0)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "duration_sec": total_duration,
        "rps": rps,
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "success_rate": success_rate,
    }

async def main():
    print("=" * 70)
    print("OpenJEV SLA & Concurrency Benchmark (IMPLEMENTATION.md §1 & §9)")
    print("=" * 70)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)) as client:
        # Check server readiness
        try:
            r = await client.get("http://127.0.0.1:8000/readyz")
            if r.status_code != 200:
                print("Server is not ready. Aborting benchmark.")
                return
        except Exception as e:
            print(f"Cannot connect to server at 127.0.0.1:8000: {e}")
            return

        concurrency_levels = [1, 4, 16, 64]
        seq_lengths = [256, 512, 1024]

        results = []
        for seq_len in seq_lengths:
            payload = build_payload(seq_len)
            print(f"\n--- Testing Sequence Length: {seq_len} tokens ---")
            for c in concurrency_levels:
                num_reqs = max(c * 5, 20)
                # Warmup
                await benchmark_client(client, payload, total_requests=5, concurrency=1)
                res = await benchmark_client(client, payload, total_requests=num_reqs, concurrency=c)
                res["seq_len"] = seq_len
                results.append(res)
                print(
                    f"Concurrency: {c:2d} | Req/s: {res['rps']:6.1f} | "
                    f"p50: {res['p50_ms']:6.2f} ms | p90: {res['p90_ms']:6.2f} ms | "
                    f"p99: {res['p99_ms']:6.2f} ms | Success: {res['success_rate']:5.1f}%"
                )

        print("\n" + "=" * 70)
        print("Summary SLA Certification Table:")
        print("SeqLen | Concurrency | Req/sec | Latency p50 | Latency p90 | Latency p99")
        print("-" * 70)
        for r in results:
            print(
                f"{r['seq_len']:6d} | {r['concurrency']:11d} | {r['rps']:7.1f} | "
                f"{r['p50_ms']:9.2f} ms | {r['p90_ms']:9.2f} ms | {r['p99_ms']:9.2f} ms"
            )

if __name__ == "__main__":
    asyncio.run(main())
