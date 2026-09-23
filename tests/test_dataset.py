"""
Phase 2 Unit Test Suite for Multi-Task Dataset Builder & Synthetic Generator
Verifies:
1. NLI to Bool example conversion formatting and target probabilities.
2. Intent to Choice example conversion and option index alignment.
3. Rating to Score ordinal example mapping.
4. Batcher tensor shapes for MLX training pipelines.
5. Synthetic boundary generator output integrity & zero-ToS provenance tagging.
"""

import pytest
import mlx.core as mx
from data.dataset_builder import (
    format_nli_to_bool_example,
    format_intent_to_choice_example,
    format_rating_to_score_example,
    DatasetBatcher
)
from data.synthetic_generator import SyntheticBoundaryGenerator

def test_nli_to_bool_formatting():
    entailment_ex = format_nli_to_bool_example("The sky is blue.", "It is daytime.", "entailment")
    assert entailment_ex["primitive"] == "bool"
    assert entailment_ex["target_prob"] == 1.0
    assert entailment_ex["target_value"] is True

    contradiction_ex = format_nli_to_bool_example("The cat is asleep.", "The cat is running.", "contradiction")
    assert contradiction_ex["target_prob"] == 0.0
    assert contradiction_ex["target_value"] is False

    neutral_ex = format_nli_to_bool_example("A man is reading.", "He is reading a mystery novel.", "neutral")
    assert neutral_ex["target_prob"] == 0.5

def test_intent_to_choice_formatting():
    intents = [
        ("billing", "Invoice and payment inquiries."),
        ("support", "Technical issue troubleshooting.")
    ]
    ex = format_intent_to_choice_example("My card was charged twice.", intents, "billing")
    
    assert ex["primitive"] == "choice"
    assert len(ex["options"]) == 2
    assert ex["target_idx"] == 0

def test_rating_to_score_formatting():
    ex = format_rating_to_score_example("Great product!", "1-5 scale rating", 4, min_val=1, max_val=5)
    
    assert ex["primitive"] == "score"
    assert ex["target_rating"] == 4
    assert ex["target_level_idx"] == 3 # 4 - 1 = 3

def test_dataset_batcher_shapes():
    bool_examples = [
        format_nli_to_bool_example("P1", "H1", "entailment"),
        format_nli_to_bool_example("P2", "H2", "contradiction")
    ]
    h_q_bool, targets_bool = DatasetBatcher.batch_bool_queries(bool_examples, hidden_dim=32)
    mx.eval(h_q_bool, targets_bool)
    
    assert h_q_bool.shape == (2, 32)
    assert targets_bool.shape == (2,)

    intents = [("id1", "desc1"), ("id2", "desc2")]
    choice_examples = [
        format_intent_to_choice_example("text1", intents, "id1"),
        format_intent_to_choice_example("text2", intents, "id2")
    ]
    h_q_choice, e_options, target_indices = DatasetBatcher.batch_choice_queries(choice_examples, hidden_dim=32)
    mx.eval(h_q_choice, e_options, target_indices)
    
    assert h_q_choice.shape == (2, 32)
    assert e_options.shape == (2, 2, 32)
    assert target_indices.shape == (2,)

def test_synthetic_boundary_generator():
    gen = SyntheticBoundaryGenerator(seed=123)
    bool_set = gen.generate_bool_boundary_set(count=10)
    choice_set = gen.generate_choice_routing_set(count=10)
    
    assert len(bool_set) == 10
    assert len(choice_set) == 10
    assert bool_set[0]["provenance"] == "rule_based_synthetic_v1"
    assert choice_set[0]["provenance"] == "rule_based_synthetic_v1"
