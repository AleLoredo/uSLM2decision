"""
OpenJEV Multi-Task Balanced Dataset Generator (System 1)
Generates high-quality balanced datasets for OpenJEV's three canonical primitives:
1. `noul` (Boolean truth/urgency evaluation: targets 0 or 1)
2. `choice` (Categorical routing over criteria map: target option ID)
3. `score` (Ordinal rubric scoring: target discrete level integer)

Stores results in:
  ./train-dataset/
  ├── train.jsonl
  ├── val.jsonl
  ├── test.jsonl
  └── dataset_card.json
"""

import os
import json
import random
from typing import Dict, List, Any

# Ensure reproducible generation
RNG = random.Random(42)

DATASET_DIR = "train-dataset"

# ---------------------------------------------------------
# 1. NOUL (Boolean / Is this true? / Urgency / Cancellation)
# ---------------------------------------------------------

NOUL_URGENT_POSITIVES = [
    "you idiots i need this fixed by today",
    "our payment gateway is returning 500 errors on every request, we cannot process customer orders",
    "emergency: production database is locked and customer checkout is completely down",
    "I am losing sales right now, please respond immediately ASAP",
    "my account was compromised and unauthorized charges are occurring right now",
    "critical bug in production API affecting all enterprise customers",
    "we are in the middle of our product launch and the integration is failing, urgent help needed",
    "system outage: all webhooks stopped firing 15 minutes ago, critical priority",
    "help! our entire team is locked out of the dashboard during customer hours",
    "billing double charged our credit card 10 times in 5 minutes, stop this right now"
]

NOUL_URGENT_NEGATIVES = [
    "hi, just wondering if you support dark mode on mobile?",
    "no rush at all, just wanted to ask when the next feature release is planned",
    "great product! one small typo in the documentation on page 3",
    "when you have time next week, could you clarify your pricing tiers?",
    "inquiring about whether you have an affiliate partnership program",
    "just sharing some feedback, everything is working smoothly",
    "curious if you will support Python 3.14 in the upcoming quarter",
    "hello team, hope you are having a nice weekend, quick question about receipt formatting",
    "minor suggestion: it would be nice to have a search shortcut",
    "everything is working fine, just wanted to confirm my renewal date next month"
]

NOUL_CANCEL_POSITIVES = [
    "please cancel my subscription immediately, I do not want to be charged again",
    "I want to terminate my account and delete all associated data",
    "cancel my plan right now and issue a refund for this month",
    "close my account, I am switching to another provider",
    "stop auto-renewal and cancel my membership"
]

NOUL_CANCEL_NEGATIVES = [
    "how do I upgrade my account to the enterprise plan?",
    "we would like to add 15 more user seats to our current subscription",
    "can we renew our annual contract for another two years?",
    "where can I download the invoice for our renewed subscription?",
    "we love the tool and want to expand it to our entire engineering team"
]

# ---------------------------------------------------------
# 2. CHOICE (Which of these options?)
# ---------------------------------------------------------

CHOICE_CRITERIA_DEPARTMENTS = {
    "billing": "Payment, subscription, invoice or charge issues",
    "technical": "API errors, integration bugs, code crashes or system outages",
    "sales": "Pricing inquiries, custom enterprise plans, or quotes"
}

CHOICE_SAMPLES_DEPARTMENT = [
    # Billing
    ("there is a problem with my billling. you idiots i need this fixed by today", "billing"),
    ("My credit card was charged twice for the monthly subscription.", "billing"),
    ("Can I get an updated VAT invoice for our accounting team?", "billing"),
    ("Where can I update our company bank details and payment method?", "billing"),
    ("Why did the subscription renew at a higher rate without notice?", "billing"),
    ("We need a refund for the unused seats from last quarter.", "billing"),
    ("Payment failed with error card_declined, how do we retry?", "billing"),
    ("Our account shows past due even though we wired payment yesterday.", "billing"),
    
    # Technical
    ("Our API integration started returning 500 errors on every request 20 minutes ago.", "technical"),
    ("The webhook endpoint is throwing 502 Bad Gateway intermittently.", "technical"),
    ("I am getting a TypeError: NoneType object is not callable in the Python SDK.", "technical"),
    ("Docker container exits immediately with exit code 137 OOM killed.", "technical"),
    ("How do I configure mutual TLS for the websocket connection?", "technical"),
    ("The database migration script failed on alter table command.", "technical"),
    ("Requests are timing out after 30 seconds when querying the search API.", "technical"),
    ("Our SSL certificate seems to have expired on the API gateway.", "technical"),

    # Sales
    ("What are your enterprise pricing tiers for 250 active agents?", "sales"),
    ("We would like to request a formal RFP quote for our annual procurement.", "sales"),
    ("Do you offer volume discounts for non-profit organizations or universities?", "sales"),
    ("We want to schedule a product demo with our VP of Engineering.", "sales"),
    ("Is there an annual discount if we pay upfront instead of monthly?", "sales"),
    ("Can we get custom terms and an SLA guarantee for enterprise tier?", "sales"),
    ("How does your pricing compare with competitor X for 5 million tokens?", "sales"),
    ("We want to discuss custom contract terms before signing the order form.", "sales")
]

CHOICE_CRITERIA_PRIORITY = {
    "p0_critical": "System completely down, severe revenue or data loss",
    "p1_high": "Major functionality broken with no workaround",
    "p2_medium": "Minor feature impaired, workaround available",
    "p3_low": "Cosmetic issue, feature request, or general question"
}

CHOICE_SAMPLES_PRIORITY = [
    ("All customer payments are failing in production right now!", "p0_critical"),
    ("Core service is crashing on every request across all regions.", "p0_critical"),
    ("Export to CSV is failing, but users can still view data on screen.", "p1_high"),
    ("Single user cannot reset password, but admin can reset it manually.", "p2_medium"),
    ("Button padding looks slightly misaligned on Safari browser.", "p3_low"),
    ("Would love to see a dark mode option added in the future.", "p3_low")
]

# ---------------------------------------------------------
# 3. SCORE (Which level on rubric?)
# ---------------------------------------------------------

SCORE_RUBRIC_FRUSTRATION = [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language"
]

SCORE_SAMPLES_FRUSTRATION = [
    # Level 0 (Calm)
    ("Hello, could you please clarify how the API rate limit is counted?", 0),
    ("Hi, our team is testing the endpoint and we have a quick question.", 0),
    ("Good morning, here is the log snippet for your review when convenient.", 0),
    ("I noticed an inconsistency in the documentation for the decide endpoint.", 0),
    ("Thank you for the quick reply, could you also confirm the port number?", 0),

    # Level 1 (Frustrated)
    ("I have been trying to connect my Stripe account for 3 days and the integration keeps failing.", 1),
    ("This is the second time this week our sync has broken, please look into this.", 1),
    ("I submitted a ticket 48 hours ago and have not received any response yet.", 1),
    ("The error messages are completely unhelpful and we are losing time.", 1),
    ("We followed the docs step by step and it still does not work as described.", 1),

    # Level 2 (Very angry)
    ("there is a problem with my billling. you idiots i need this fixed by today", 2),
    ("this is completely unacceptable and your software is garbage!", 2),
    ("you charged us 10 times and your support is utterly useless, fix this immediately!", 2),
    ("what a joke of a company, your incompetence is costing us thousands of dollars!", 2),
    ("I want to speak with your manager immediately, this is atrocious service!", 2)
]

SCORE_RUBRIC_SEVERITY = [
    "Informational or trivial",
    "Minor impairment",
    "Severe disruption"
]

SCORE_SAMPLES_SEVERITY = [
    ("Typo on the landing page header text.", 0),
    ("Icon color is slightly off in dark mode.", 0),
    ("Notification email was delayed by 5 minutes.", 1),
    ("Search filters take 2 seconds longer than usual.", 1),
    ("Entire customer database is offline and requests are failing 100%.", 2),
    ("Security breach: customer tokens leaked in public logs.", 2)
]

# ---------------------------------------------------------
# Generator Functions with Data Augmentation
# ---------------------------------------------------------

def generate_noul_dataset(target_count: int = 500) -> List[Dict[str, Any]]:
    samples = []
    
    # 1. Urgency evaluation
    q_urg = {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity"
    }
    for _ in range(target_count // 4):
        p = RNG.choice(NOUL_URGENT_POSITIVES)
        samples.append({
            "primitive": "noul",
            "state": p,
            "question": q_urg,
            "target": 1
        })
        n = RNG.choice(NOUL_URGENT_NEGATIVES)
        samples.append({
            "primitive": "noul",
            "state": n,
            "question": q_urg,
            "target": 0
        })

    # 2. Cancellation evaluation
    q_canc = {
        "type": "noul",
        "instructions": "The user is requesting account or subscription cancellation"
    }
    for _ in range(target_count // 4):
        p = RNG.choice(NOUL_CANCEL_POSITIVES)
        samples.append({
            "primitive": "noul",
            "state": p,
            "question": q_canc,
            "target": 1
        })
        n = RNG.choice(NOUL_CANCEL_NEGATIVES)
        samples.append({
            "primitive": "noul",
            "state": n,
            "question": q_canc,
            "target": 0
        })
        
    return samples

def generate_choice_dataset(target_count: int = 500) -> List[Dict[str, Any]]:
    samples = []
    
    # 1. Department routing
    q_dept = {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": CHOICE_CRITERIA_DEPARTMENTS
    }
    for _ in range(target_count // 2):
        text, target = RNG.choice(CHOICE_SAMPLES_DEPARTMENT)
        samples.append({
            "primitive": "choice",
            "state": text,
            "question": q_dept,
            "target": target
        })

    # 2. Priority routing
    q_prio = {
        "type": "choice",
        "instructions": "Select the appropriate incident severity priority",
        "criteria": CHOICE_CRITERIA_PRIORITY
    }
    for _ in range(target_count // 2):
        text, target = RNG.choice(CHOICE_SAMPLES_PRIORITY)
        samples.append({
            "primitive": "choice",
            "state": text,
            "question": q_prio,
            "target": target
        })
        
    return samples

def generate_score_dataset(target_count: int = 500) -> List[Dict[str, Any]]:
    samples = []
    
    # 1. Frustration rubric
    q_frust = {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": SCORE_RUBRIC_FRUSTRATION
    }
    for _ in range(target_count // 2):
        text, target = RNG.choice(SCORE_SAMPLES_FRUSTRATION)
        samples.append({
            "primitive": "score",
            "state": text,
            "question": q_frust,
            "target": target
        })

    # 2. Severity rubric
    q_sev = {
        "type": "score",
        "instructions": "Rate the operational impact severity level",
        "criteria": SCORE_RUBRIC_SEVERITY
    }
    for _ in range(target_count // 2):
        text, target = RNG.choice(SCORE_SAMPLES_SEVERITY)
        samples.append({
            "primitive": "score",
            "state": text,
            "question": q_sev,
            "target": target
        })
        
    return samples

def build_all_training_data(output_dir: str = DATASET_DIR, count_per_primitive: int = 400) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    
    noul_data = generate_noul_dataset(count_per_primitive)
    choice_data = generate_choice_dataset(count_per_primitive)
    score_data = generate_score_dataset(count_per_primitive)
    
    all_records = noul_data + choice_data + score_data
    RNG.shuffle(all_records)
    
    total = len(all_records)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)
    
    train_records = all_records[:train_end]
    val_records = all_records[train_end:val_end]
    test_records = all_records[val_end:]
    
    def write_jsonl(path: str, records: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                item = {"id": f"rec_{i:05d}", **r}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)
    
    manifest = {
        "dataset_name": "openjev-system1-multitask",
        "version": "1.0.0",
        "output_directory": output_dir,
        "total_samples": total,
        "split": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "primitives": {
            "noul": len(noul_data),
            "choice": len(choice_data),
            "score": len(score_data)
        },
        "license": "Apache-2.0"
    }
    
    card_path = os.path.join(output_dir, "dataset_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✅ Dataset multi-tarea generado exitosamente en: {output_dir}/")
    print(f"   Total: {total} ejemplos (Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)})")
    print(f"   Primitivas balanceadas: noul={len(noul_data)}, choice={len(choice_data)}, score={len(score_data)}")
    return manifest

if __name__ == "__main__":
    build_all_training_data()
