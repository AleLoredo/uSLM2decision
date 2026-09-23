"""
Rule-Based Synthetic Boundary Data Generator
Generates synthetic boundary queries, edge-case negations, and ambiguous decision pairs
without relying on proprietary API models (Zero ToS License Liability).
"""

from typing import Dict, List, Any
import random

class SyntheticBoundaryGenerator:
    """
    Deterministic rule-based generator for edge cases and boundary queries.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates_bool = [
            ("The document states '{topic} is active'.", "Is {topic} enabled?", True),
            ("The document states '{topic} is inactive'.", "Is {topic} enabled?", False),
            ("The document explicitly forbids {topic}.", "Is {topic} allowed?", False),
            ("The document permits {topic} under condition X.", "Is {topic} allowed?", True),
            ("There is no mention of {topic} in the text.", "Is {topic} mentioned?", False),
        ]
        self.topics = ["authentication", "encryption", "rate_limiting", "caching", "telemetry", "logging"]

    def generate_bool_boundary_set(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generates rule-based bool boundary queries.
        """
        results = []
        for i in range(count):
            tmpl, criteria, is_true = random.choice(self.templates_bool)
            topic = random.choice(self.topics)
            context = tmpl.format(topic=topic)
            crit = criteria.format(topic=topic)
            
            results.append({
                "primitive": "bool",
                "context": context,
                "criteria": crit,
                "target_prob": 1.0 if is_true else 0.0,
                "target_value": is_true,
                "provenance": "rule_based_synthetic_v1"
            })
        return results

    def generate_choice_routing_set(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generates rule-based intent routing choice queries.
        """
        intents = [
            ("billing", "Questions regarding invoices, payment methods, or charges."),
            ("tech_support", "Issues with API errors, system crashes, or integration bugs."),
            ("account", "Account settings, password reset, or profile updates."),
            ("general", "General inquiries about product features or pricing.")
        ]
        
        utterances = [
            ("Where can I download my latest PDF invoice?", "billing"),
            ("I am receiving HTTP 500 Internal Server Error when calling endpoint.", "tech_support"),
            ("How do I update my registered email address?", "account"),
            ("What are your enterprise subscription tiers?", "general")
        ]
        
        results = []
        for i in range(count):
            utt, correct_id = random.choice(utterances)
            target_idx = next(idx for idx, (i_id, _) in enumerate(intents) if i_id == correct_id)
            
            options = [{"id": i_id, "description": desc} for i_id, desc in intents]
            
            results.append({
                "primitive": "choice",
                "context": utt,
                "criteria": "Select the relevant support department.",
                "options": options,
                "target_idx": target_idx,
                "provenance": "rule_based_synthetic_v1"
            })
        return results
