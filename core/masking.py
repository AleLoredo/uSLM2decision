"""
OpenJEV Block-Causal Masking and Position-ID Generator
Implements ARCHITECTURE.md §2:
- Additive Block-Causal Attention Mask (Queries attend to State and self causally, never across queries).
- Per-Segment Position-ID Resets (Each query starts at position S, eliminating sequence order bias).
"""

import numpy as np
import mlx.core as mx

def create_block_causal_mask(s_len: int, q_lens: list[int], state_causal: bool = True) -> mx.array:
    """
    Constructs additive attention mask [1, 1, Total, Total].
    0.0 = attend, -1e9 = masked.
    Layout: [STATE (s_len)] [QUERY_1 (q_lens[0])] ... [QUERY_K (q_lens[K-1])]
    """
    total = s_len + sum(q_lens)
    mask = np.full((total, total), -1e9, dtype=np.float32)
    
    # 1. State tokens
    if state_causal:
        for i in range(s_len):
            for j in range(i + 1):
                mask[i, j] = 0.0
    else:
        # Full bidirectional state
        mask[:s_len, :s_len] = 0.0
        
    # 2. Query tokens
    current_q_offset = s_len
    for q_len in q_lens:
        q_start = current_q_offset
        q_end = current_q_offset + q_len
        for i in range(q_start, q_end):
            # Query attends to ALL state tokens (0..s_len)
            mask[i, :s_len] = 0.0
            # Query attends to itself causally (q_start..=i)
            mask[i, q_start:i + 1] = 0.0
        current_q_offset += q_len
        
    return mx.array(mask).reshape(1, 1, total, total)

def create_segment_position_ids(s_len: int, q_lens: list[int]) -> mx.array:
    """
    Constructs per-segment position IDs:
    State: 0 .. S-1
    Query_k: S .. S + Q_k - 1 (resets to S for EVERY query block)
    """
    pos_ids = list(range(s_len))
    for q_len in q_lens:
        for i in range(q_len):
            pos_ids.append(s_len + i)
    return mx.array(pos_ids, dtype=mx.int32)
