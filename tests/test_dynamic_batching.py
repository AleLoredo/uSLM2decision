"""
Unit and Concurrency Tests for Dynamic Micro-Batching (IMPLEMENTATION.md §4)
Verifies:
1. BatchQueue accumulation up to max_batch_size.
2. BatchQueue timeout expiration after max_wait_ms.
3. Queue depth bounding and QueueFullError backpressure.
4. Multi-request sequence packing with strict block-causal mathematical isolation.
5. BatchDispatcher concurrent execution across multiple requests without crosstalk.
6. Server-level concurrent load handling.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.batching.types import QueuedRequest, QueueFullError
from core.batching.queue import BatchQueue
from core.batching.packer import pack_requests, extract_question_text
from core.batching.dispatcher import BatchDispatcher
from core.config import load_config
from core.engine_factory import create_engine
from server.edge_server import app

def test_batch_queue_accumulation():
    queue = BatchQueue(max_batch_size=4, max_wait_ms=50, max_queue_depth=10)
    for i in range(4):
        queue.put(QueuedRequest(request_id=f"r_{i}", state="ctx", questions={}))

    batch = queue.get_batch(timeout=0.1)
    assert len(batch) == 4
    assert [r.request_id for r in batch] == ["r_0", "r_1", "r_2", "r_3"]

def test_batch_queue_timeout_partial_batch():
    queue = BatchQueue(max_batch_size=8, max_wait_ms=15, max_queue_depth=10)
    # Put only 2 requests
    queue.put(QueuedRequest(request_id="r_0", state="ctx", questions={}))
    queue.put(QueuedRequest(request_id="r_1", state="ctx", questions={}))

    t0 = time.time()
    batch = queue.get_batch(timeout=0.1)
    duration = time.time() - t0

    assert len(batch) == 2
    assert [r.request_id for r in batch] == ["r_0", "r_1"]
    # Should have waited for deadline (~15ms)
    assert duration >= 0.01

def test_batch_queue_backpressure_rejection():
    queue = BatchQueue(max_batch_size=4, max_wait_ms=10, max_queue_depth=3)
    for i in range(3):
        queue.put(QueuedRequest(request_id=f"r_{i}", state="ctx", questions={}))

    with pytest.raises(QueueFullError):
        queue.put(QueuedRequest(request_id="r_overflow", state="ctx", questions={}))

def test_multi_request_packer_isolation():
    def mock_tokenizer(text: str):
        return [abs(hash(w)) % 1000 + 1 for w in text.split()]

    req1 = QueuedRequest(
        request_id="req_1",
        state="Alpha state document", # 3 tokens
        questions={
            "q1_1": {"type": "noul", "instructions": "Is alpha true?"} # 3 tokens
        }
    )
    req2 = QueuedRequest(
        request_id="req_2",
        state="Beta context information here", # 4 tokens
        questions={
            "q2_1": {"type": "noul", "instructions": "Is beta active?"} # 3 tokens
        }
    )

    packed = pack_requests([req1, req2], tokenize_fn=mock_tokenizer, state_causal=True)
    mask_np = np.array(packed.mask)[0, 0] # (T, T)
    pos_ids_np = np.array(packed.position_ids)[0] # (T,)

    len_req1 = len(mock_tokenizer(req1.state)) + len(mock_tokenizer(req1.questions["q1_1"]["instructions"]))
    len_req2 = len(mock_tokenizer(req2.state)) + len(mock_tokenizer(req2.questions["q2_1"]["instructions"]))
    total_tokens = len_req1 + len_req2
    assert packed.total_tokens == total_tokens
    assert mask_np.shape == (total_tokens, total_tokens)

    # 1. Check strict cross-request isolation (-1e9)
    # Tokens from req1 looking at req2
    assert np.all(mask_np[:len_req1, len_req1:] <= -1e8)
    # Tokens from req2 looking at req1
    assert np.all(mask_np[len_req1:, :len_req1] <= -1e8)

    # 2. Check intra-request state and causal attention for req1
    # Req1 query (tokens 3..5) attends to Req1 state (tokens 0..2)
    assert np.all(mask_np[3:len_req1, :3] == 0.0)

    # 3. Check intra-request state and causal attention for req2
    # Req2 query attends to Req2 state
    req2_q_start = len_req1 + 4
    req2_s_slice = slice(len_req1, len_req1 + 4)
    assert np.all(mask_np[req2_q_start:, req2_s_slice] == 0.0)

    # 4. Check position IDs reset at each request state
    assert pos_ids_np[0] == 0 # req1 state start
    assert pos_ids_np[len_req1] == 0 # req2 state starts at 0!

@pytest.mark.anyio
async def test_batch_dispatcher_concurrent_execution():
    cfg = load_config()
    engine = create_engine(cfg)
    engine.load_model()

    queue = BatchQueue(max_batch_size=8, max_wait_ms=10, max_queue_depth=100)
    dispatcher = BatchDispatcher(engine=engine, queue=queue)
    dispatcher.start()

    loop = asyncio.get_running_loop()

    async def send_single_request(idx: int):
        fut = loop.create_future()
        req = QueuedRequest(
            request_id=f"concurrent_{idx}",
            state=f"Context for worker request number {idx}",
            questions={
                f"q_noul_{idx}": {"type": "noul", "instructions": f"Is condition {idx} satisfied?"},
                f"q_choice_{idx}": {
                    "type": "choice",
                    "instructions": "Select option",
                    "criteria": {"a": "Opt A", "b": "Opt B"}
                }
            },
            loop=loop,
            future=fut
        )
        queue.put(req)
        return await fut

    # Send 8 concurrent requests
    tasks = [send_single_request(i) for i in range(8)]
    results = await asyncio.gather(*tasks)

    dispatcher.stop()

    assert len(results) == 8
    for i, res in enumerate(results):
        answers = res["answers"]
        assert f"q_noul_{i}" in answers
        assert answers[f"q_noul_{i}"]["type"] == "noul"
        assert answers[f"q_noul_{i}"]["noul"] in [0, 1]
        assert f"q_choice_{i}" in answers
        assert answers[f"q_choice_{i}"]["type"] == "choice"

def test_server_concurrent_requests_under_batching():
    client = TestClient(app)

    def post_request(idx: int):
        payload = {
            "state": f"Document context payload for client thread {idx}",
            "questions": {
                f"q_{idx}": {"type": "noul", "instructions": f"Criterion check {idx}"}
            }
        }
        res = client.post("/v1/decide", json=payload)
        return res.status_code, res.json()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(post_request, i) for i in range(12)]
        results = [f.result() for f in futures]

    assert len(results) == 12
    for status_code, data in results:
        assert status_code == 200
        assert "answers" in data
        assert len(data["answers"]) == 1
        q_ans = list(data["answers"].values())[0]
        assert q_ans["type"] == "noul"
        assert q_ans["noul"] in [0, 1]
