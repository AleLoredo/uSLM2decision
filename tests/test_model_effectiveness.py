"""
Model Effectiveness & Generalization Benchmark Test (SPEC.md §8 & ARCHITECTURE.md §8)
Evaluates the real inference engine on the blind held-out test partition (`train-dataset/test.jsonl`).
Verifies:
1. Noul boolean classification accuracy >= 95% on unseen data.
2. Choice routing accuracy >= 95% on unseen data.
3. Score ordinal accuracy (within +/- 1 rubric level) >= 95% with zero severe distance errors (>= 2 levels).
"""

import os
import json
import pytest
from core.config import load_config
from core.engine_factory import create_engine

DATASET_PATH = "train-dataset/test.jsonl"

@pytest.fixture(scope="module")
def initialized_engine():
    config = load_config()
    engine = create_engine(config)
    engine.load_model()
    return engine

def test_held_out_dataset_effectiveness(initialized_engine):
    assert os.path.isfile(DATASET_PATH), f"Held-out test dataset not found at {DATASET_PATH}"
    
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    assert len(records) > 0, "Test dataset is empty"
    
    noul_correct = 0
    noul_total = 0
    choice_correct = 0
    choice_total = 0
    score_exact = 0
    score_near = 0
    score_severe_errors = 0
    score_total = 0

    for r in records:
        prim = r["primitive"]
        state = r["state"]
        q = r["question"]
        target = r["target"]
        
        res = initialized_engine.decide(state, {"q": q})
        ans = res["answers"]["q"]
        
        if prim == "noul":
            pred = ans["noul"]
            if pred == int(target):
                noul_correct += 1
            noul_total += 1
            
        elif prim == "choice":
            pred = ans["choice"]
            if pred == target:
                choice_correct += 1
            choice_total += 1
            
        elif prim == "score":
            pred = int(round(ans["score"]))
            diff = abs(pred - int(target))
            if diff == 0:
                score_exact += 1
                score_near += 1
            elif diff == 1:
                score_near += 1
            else:
                score_severe_errors += 1
            score_total += 1

    # Assertions on held-out generalizations
    noul_acc = noul_correct / noul_total
    choice_acc = choice_correct / choice_total
    score_near_acc = score_near / score_total
    
    print(f"\n[Held-out Test Evaluation]")
    print(f"  Noul Accuracy:       {noul_acc * 100:.2f}% ({noul_correct}/{noul_total})")
    print(f"  Choice Accuracy:     {choice_acc * 100:.2f}% ({choice_correct}/{choice_total})")
    print(f"  Score Exact Match:   {score_exact / score_total * 100:.2f}% ({score_exact}/{score_total})")
    print(f"  Score (+/- 1 Level): {score_near_acc * 100:.2f}% ({score_near}/{score_total})")
    print(f"  Score Severe Misses: {score_severe_errors}/{score_total}")

    assert noul_acc >= 0.95, f"Noul generalization failed: {noul_acc:.2%}"
    assert choice_acc >= 0.95, f"Choice generalization failed: {choice_acc:.2%}"
    assert score_near_acc >= 0.95, f"Score ordinal near-miss bound failed: {score_near_acc:.2%}"
    assert score_severe_errors == 0, f"Found {score_severe_errors} severe ordinal errors (distance >= 2)"

def test_polite_commercial_query_frustration_regression(initialized_engine):
    """
    Regression test for commercial query emotion decoupling:
    Ensures that polite subscription / pricing / cost inquiries are correctly
    classified as Level 0 (Calm), not conflated with Level 2 anger/profanity.
    """
    state = "would it be possible to eventually reduce the cost of my active subscription to your service?"
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions"
            }
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language"
            ]
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity"
        }
    }
    
    res = initialized_engine.decide(state, questions)
    answers = res["answers"]
    
    # 1. Choice should route to sales
    assert answers["department"]["choice"] == "sales"
    assert answers["department"]["confidence"] >= 0.80
    
    # 2. Noul should indicate non-urgent (0)
    assert answers["is_urgent"]["noul"] == 0
    
    # 3. Score must be Level 0 (Calm, not Angry Level 2)
    frust = answers["frustration"]
    probs = frust["probabilities"]
    modal_level = max(probs, key=probs.get)
    assert modal_level == "0", f"Expected modal level 0 (Calm), got {modal_level} with probs {probs}"
    assert frust["score"] < 0.5, f"Expected score < 0.5 (Calm), got {frust['score']}"
    assert probs["0"] > 0.60, f"Expected Calm probability > 60%, got {probs['0']}"
    assert probs["2"] < 0.15, f"Expected Angry probability < 15%, got {probs['2']}"

