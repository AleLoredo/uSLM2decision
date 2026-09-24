# DATASET_CARD.md: Provenance & Licensing for OpenJEV Tuning Corpora

- **Dataset Version:** 1.0.0
- **Status:** Normative / Provenance Audited
- **License Compliance Target:** Open-Source clean release (Zero commercial ToS contamination)

---

## 1. Provenance Summary

All training and evaluation datasets used in OpenJEV multi-task supervised tuning are sourced exclusively from open-license academic corpora or generated using rule-based/open-weight models. **No output from restricted commercial API endpoints (e.g. OpenAI, Anthropic) was used to generate synthetic training data.**

| Corpus Name | Primary Task Mapping | Original Source / Author | License Type | Commercial Use Allowed? |
|---|---|---|---|---|
| **MNLI** (Multi-Genre NLI) | `bool` (entailment), `choice` | Williams et al., NYU (2018) | OANC / CC-BY 4.0 | Yes |
| **SNLI** (Stanford NLI) | `bool` (entailment) | Bowman et al., Stanford (2015) | CC-BY 4.0 | Yes |
| **ANLI** (Adversarial NLI) | `bool` boundary cases | Nie et al., UNC/Meta (2020) | CC-BY-NC 4.0 / Apache 2.0 | Research / Open Audit |
| **CLINC150** | `choice` (intent routing) | Larson et al., UMich (2019) | CC-BY 4.0 | Yes |
| **Rule-Based Synthetic** | `bool`, `choice`, `score` | OpenJEV Internal Generator | MIT / Apache 2.0 | Yes |

---

## 2. Primitive Formats & Conversion Rules

### 2.1 `bool` Primitive (NLI Entailment Mapping)
- **Context:** Premise text $P$.
- **Criteria:** Hypothesis text $H$.
- **Target:** `1.0` if label is `entailment`, `0.0` if label is `contradiction`. `neutral` examples map to boundary calibration set.

### 2.2 `choice` Primitive (Multi-Class / Intent Classification Mapping)
- **Context:** User input or document text.
- **Criteria:** "Select the matching user intent from the options."
- **Options:** Array of `{ "id": intent_id, "description": intent_label_description }`.
- **Target:** Index of ground-truth intent $C_i$.

### 2.3 `score` Primitive (Ordinal Rating Mapping)
- **Context:** Review or evaluation context.
- **Rubric:** Ordinal criteria scale description (1 = Very Poor to 5 = Excellent).
- **Target:** Discrete integer level $L \in [1, 5]$.

---

## 3. Synthetic Data Provenance Assurance

To preserve legal compliance for downstream commercial and open-source adopters:
- Synthetic edge-case queries are generated via deterministic rule-based template expansion ([`data/synthetic_generator.py`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/data/synthetic_generator.py)).
- Any model-assisted synthetic generation uses strictly open-weight models under Apache 2.0 or MIT licenses (e.g. Qwen2.5, Llama-3-Instruct open deployments).
