"""
OpenDecide Phase 0 Hardware Benchmark
Measures actual inference latency (p50, p90, p99), throughput, and memory footprint on Apple Silicon (M3).
Covers:
- Models: 0.5B (Fast tier, Qwen2.5-0.5B dims: d=896, L=24, H=14, kv_H=2)
          1.5B (Balanced tier, Qwen2.5-1.5B dims: d=1536, L=28, H=12, kv_H=2)
- Sequence Lengths: 256, 512, 1024 tokens
- Batch sizes: 1, 2, 4, 8
- Precision: FP16
"""

import time
import statistics
import mlx.core as mx
import mlx.nn as nn
import numpy as np

class TransformerLayerMock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        # Self attention block
        residual = x
        norm_x = self.norm1(x)
        B, S, _ = norm_x.shape
        
        q = self.q_proj(norm_x).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(norm_x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(norm_x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Repeat KV heads for GQA
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)
        
        # Scale dot product
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, S, self.hidden_dim)
        x = residual + self.out_proj(out)
        
        # MLP block (SwiGLU)
        residual = x
        norm_x = self.norm2(x)
        gate = nn.silu(self.gate_proj(norm_x))
        up = self.up_proj(norm_x)
        mlp_out = self.down_proj(gate * up)
        return residual + mlp_out

class DecisionModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, num_kv_heads: int, intermediate_dim: int):
        super().__init__()
        self.embed = nn.Embedding(151936, hidden_dim)
        self.layers = [
            TransformerLayerMock(hidden_dim, num_heads, num_kv_heads, intermediate_dim)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_dim)
        self.bool_head = nn.Linear(hidden_dim, 1)
        self.choice_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_head = nn.Linear(hidden_dim, 5)

    def __call__(self, input_ids: mx.array, mask: mx.array = None) -> tuple[mx.array, mx.array, mx.array]:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # Take pooled query token (last token)
        h_q = x[:, -1:, :]
        b = self.bool_head(h_q)
        c = self.choice_head(h_q)
        s = self.score_head(h_q)
        return b, c, s

def run_benchmark(tier_name: str, config: dict, seq_lens: list[int], batch_sizes: list[int], n_iters: int = 30):
    print(f"\n=======================================================")
    print(f" BENCHMARK: {tier_name} Tier ({config['num_layers']} layers, d={config['hidden_dim']})")
    print(f"=======================================================")
    
    # Initialize model
    model = DecisionModel(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        intermediate_dim=config["intermediate_dim"]
    )
    mx.eval(model.parameters())
    
    results = []
    
    for seq_len in seq_lens:
        for b_sz in batch_sizes:
            input_ids = mx.random.randint(0, 1000, shape=(b_sz, seq_len))
            # Create additive mask
            mask_np = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            mask = mx.array(mask_np).reshape(1, 1, seq_len, seq_len)
            
            # Warmup (3 runs)
            for _ in range(5):
                b, c, s = model(input_ids, mask)
                mx.eval(b, c, s)
                
            latencies = []
            for _ in range(n_iters):
                start = time.perf_counter()
                b, c, s = model(input_ids, mask)
                mx.eval(b, c, s)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0) # ms
                
            p50 = statistics.median(latencies)
            p90 = np.percentile(latencies, 90)
            p99 = np.percentile(latencies, 99)
            mean_lat = statistics.mean(latencies)
            throughput = (b_sz / (mean_lat / 1000.0))
            
            # Peak memory
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)
            
            print(f"SeqLen: {seq_len:4d} | Batch: {b_sz:2d} | p50: {p50:6.2f}ms | p90: {p90:6.2f}ms | p99: {p99:6.2f}ms | Req/s: {throughput:6.1f} | Peak VRAM: {peak_mb:.1f}MB")
            results.append({
                "tier": tier_name,
                "seq_len": seq_len,
                "batch_size": b_sz,
                "p50_ms": p50,
                "p90_ms": p90,
                "p99_ms": p99,
                "req_per_sec": throughput,
                "peak_mem_mb": peak_mb
            })
            
    return results

if __name__ == "__main__":
    # Config matching Qwen2.5-0.5B (Fast Tier)
    fast_config = {
        "hidden_dim": 896,
        "num_layers": 24,
        "num_heads": 14,
        "num_kv_heads": 2,
        "intermediate_dim": 4864
    }
    
    # Config matching Qwen2.5-1.5B (Balanced Tier)
    balanced_config = {
        "hidden_dim": 1536,
        "num_layers": 28,
        "num_heads": 12,
        "num_kv_heads": 2,
        "intermediate_dim": 8960
    }
    
    seq_lengths = [256, 512, 1024]
    batches = [1, 2, 4]
    
    print("Starting Hardware Benchmarks on Apple M3...")
    fast_results = run_benchmark("0.5B (Fast)", fast_config, seq_lengths, batches, n_iters=20)
    balanced_results = run_benchmark("1.5B (Balanced)", balanced_config, seq_lengths, batches, n_iters=15)
    
    print("\n>>> BENCHMARKS COMPLETE <<<")
