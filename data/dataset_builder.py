"""
OpenJEV Multi-Task Dataset Builder
Converts raw NLI, intent, and rating datasets into unified OpenJEV decision queries (bool, choice, score).
"""

from typing import Dict, List, Any, Tuple
import mlx.core as mx

def format_nli_to_bool_example(premise: str, hypothesis: str, label: str) -> Dict[str, Any]:
    """
    Converts an NLI premise/hypothesis pair into a `bool` decision query.
    
    label: 'entailment' -> value = True, target_prob = 1.0
           'contradiction' -> value = False, target_prob = 0.0
           'neutral' -> boundary case target_prob = 0.5
    """
    label_map = {
        "entailment": 1.0,
        "contradiction": 0.0,
        "neutral": 0.5
    }
    target_prob = label_map.get(label.lower(), 0.5)
    return {
        "primitive": "bool",
        "context": premise,
        "criteria": hypothesis,
        "target_prob": target_prob,
        "target_value": target_prob > 0.5
    }

def format_intent_to_choice_example(
    text: str,
    intents: List[Tuple[str, str]],
    correct_intent_id: str
) -> Dict[str, Any]:
    """
    Converts intent classification into a `choice` decision query.
    
    intents: List of (id, description) tuples
    correct_intent_id: ID of true intent
    """
    options = [{"id": i_id, "description": desc} for i_id, desc in intents]
    target_idx = next((i for i, (i_id, _) in enumerate(intents) if i_id == correct_intent_id), 0)
    
    return {
        "primitive": "choice",
        "context": text,
        "criteria": "Which category best describes the intent of the input text?",
        "options": options,
        "target_idx": target_idx
    }

def format_rating_to_score_example(
    text: str,
    rubric: str,
    rating: int,
    min_val: int = 1,
    max_val: int = 5
) -> Dict[str, Any]:
    """
    Converts numerical ratings into an ordinal `score` decision query.
    """
    rating_clamped = max(min_val, min(max_val, rating))
    target_level_idx = rating_clamped - min_val
    
    return {
        "primitive": "score",
        "context": text,
        "rubric": rubric,
        "min_val": min_val,
        "max_val": max_val,
        "target_level_idx": target_level_idx,
        "target_rating": rating_clamped
    }

class DatasetBatcher:
    """
    Prepares decision query outputs into structured MLX array batches for training.
    """
    @staticmethod
    def batch_bool_queries(examples: List[Dict[str, Any]], hidden_dim: int = 64) -> Tuple[mx.array, mx.array]:
        """
        Creates synthetic h_q representation tensors & target probability arrays for bool training.
        """
        targets = mx.array([ex["target_prob"] for ex in examples], dtype=mx.float32)
        # Mock pooled representations for batching validation
        h_q_batch = mx.random.normal((len(examples), hidden_dim))
        return h_q_batch, targets

    @staticmethod
    def batch_choice_queries(examples: List[Dict[str, Any]], hidden_dim: int = 64) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Creates h_q representation tensors, option embedding arrays, and target index arrays.
        """
        batch_size = len(examples)
        num_options = len(examples[0]["options"]) if examples else 2
        
        h_q_batch = mx.random.normal((batch_size, hidden_dim))
        e_options_batch = mx.random.normal((batch_size, num_options, hidden_dim))
        target_indices = mx.array([ex["target_idx"] for ex in examples], dtype=mx.int32)
        
        return h_q_batch, e_options_batch, target_indices

    @staticmethod
    def batch_score_queries(examples: List[Dict[str, Any]], hidden_dim: int = 64) -> Tuple[mx.array, mx.array]:
        """
        Creates h_q representation tensors & target level index arrays.
        """
        batch_size = len(examples)
        h_q_batch = mx.random.normal((batch_size, hidden_dim))
        target_levels = mx.array([ex["target_level_idx"] for ex in examples], dtype=mx.int32)
        
        return h_q_batch, target_levels
