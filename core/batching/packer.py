"""
OpenJEV Multi-Request Sequence Packer
Implements ARCHITECTURE.md §4 and §5:
- Combines B requests (1 <= B <= 16) into an isolated multi-request sequence.
- Construct 2D block-causal attention mask with complete inter-request isolation (-1e9 off-diagonal blocks).
- Per-segment position IDs mapped strictly within each request boundary.
"""

from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import json
import numpy as np
import mlx.core as mx

from core.batching.types import QueuedRequest

@dataclass
class QueryLocation:
    req_index: int
    query_index: int
    query_id: str
    query_dict: Dict[str, Any]
    last_token_idx: int
    token_slice: Tuple[int, int]

@dataclass
class PackedBatch:
    requests: List[QueuedRequest]
    total_tokens: int
    token_ids: mx.array
    mask: mx.array
    position_ids: mx.array
    query_locations: List[QueryLocation]

def serialize_state_to_text(state: Any) -> str:
    """Serializes state (string or structured JSON) to text for tokenization."""
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, indent=2)
    return str(state)

def extract_question_text(q: Dict[str, Any]) -> str:
    """Extracts text representation from question dictionary."""
    instructions = q.get("instructions", "")
    q_type = q.get("type")
    if q_type == "noul":
        crit = q.get("criteria", "")
        if crit:
            return f"{instructions} {crit}".strip()
        return instructions
    return instructions

def pack_requests(
    requests: List[QueuedRequest],
    tokenize_fn: Callable[[str], List[int]],
    state_causal: bool = True
) -> PackedBatch:
    """
    Packs multiple QueuedRequests into a single multi-sequence representation
    with isolated block-causal attention mask and per-request position IDs.
    """
    if not requests:
        raise ValueError("Cannot pack an empty list of requests.")

    all_tokens: List[int] = []
    all_pos_ids: List[int] = []
    query_locations: List[QueryLocation] = []

    req_meta = []
    global_offset = 0

    for req_idx, req in enumerate(requests):
        s_text = serialize_state_to_text(req.state)
        s_tokens = tokenize_fn(s_text)
        s_len = max(len(s_tokens), 1)

        # Questions is a Dict[str, Dict[str, Any]]
        questions_items = list(req.questions.items()) if isinstance(req.questions, dict) else [
            (q.get("id", f"q_{i}"), q) for i, q in enumerate(req.questions)
        ]

        q_tokens_list = [tokenize_fn(extract_question_text(q_dict)) for _, q_dict in questions_items]
        q_lens = [max(len(qt), 1) for qt in q_tokens_list]
        req_total = s_len + sum(q_lens)

        req_meta.append({
            "req_idx": req_idx,
            "req": req,
            "s_len": s_len,
            "q_lens": q_lens,
            "start": global_offset,
            "end": global_offset + req_total,
            "s_tokens": s_tokens or [1],
            "q_tokens_list": [qt or [1] for qt in q_tokens_list]
        })

        # Append state tokens & position IDs
        all_tokens.extend(s_tokens or [1])
        all_pos_ids.extend(range(s_len))

        # Append questions tokens & position IDs
        current_q_start = global_offset + s_len
        for q_idx, ((q_id, q_dict), q_tokens, q_len) in enumerate(zip(questions_items, q_tokens_list, q_lens)):
            actual_q_tokens = q_tokens or [1]
            all_tokens.extend(actual_q_tokens)
            for i in range(q_len):
                all_pos_ids.append(s_len + i)

            last_idx = current_q_start + q_len - 1
            query_locations.append(QueryLocation(
                req_index=req_idx,
                query_index=q_idx,
                query_id=q_id,
                query_dict=q_dict,
                last_token_idx=last_idx,
                token_slice=(current_q_start, current_q_start + q_len)
            ))
            current_q_start += q_len

        global_offset += req_total

    total_tokens = global_offset
    # Initialize mask to strictly isolated -1e9 everywhere
    mask_np = np.full((total_tokens, total_tokens), -1e9, dtype=np.float32)

    # Fill intra-request attention blocks
    for meta in req_meta:
        start = meta["start"]
        s_len = meta["s_len"]
        q_lens = meta["q_lens"]

        # 1. State tokens for this request
        if state_causal:
            for i in range(s_len):
                for j in range(i + 1):
                    mask_np[start + i, start + j] = 0.0
        else:
            mask_np[start:start + s_len, start:start + s_len] = 0.0

        # 2. Query tokens for this request
        curr_q = start + s_len
        for q_len in q_lens:
            q_start = curr_q
            q_end = curr_q + q_len
            for i in range(q_start, q_end):
                # Query attends to ALL state tokens of THIS request
                mask_np[i, start:start + s_len] = 0.0
                # Query attends to itself causally
                mask_np[i, q_start:i + 1] = 0.0
            curr_q += q_len

    token_ids_mx = mx.array(all_tokens, dtype=mx.int32).reshape(1, total_tokens)
    position_ids_mx = mx.array(all_pos_ids, dtype=mx.int32).reshape(1, total_tokens)
    mask_mx = mx.array(mask_np, dtype=mx.float32).reshape(1, 1, total_tokens, total_tokens)

    return PackedBatch(
        requests=requests,
        total_tokens=total_tokens,
        token_ids=token_ids_mx,
        mask=mask_mx,
        position_ids=position_ids_mx,
        query_locations=query_locations
    )
