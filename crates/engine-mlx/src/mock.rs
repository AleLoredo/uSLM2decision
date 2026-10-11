//! Deterministic Calibrated Mock Engine for testing and CI.

use crate::types::{EngineError, InferenceEngine};
use openjev_calibration::temperature::{scaled_sigmoid, scaled_softmax};
use openjev_core::types::{Answer, Question, UsageInfo};
use std::collections::HashMap;

pub struct MockEngine {
    model_version: String,
    calibration_version: String,
}

impl MockEngine {
    pub fn new(model_version: &str, calibration_version: &str) -> Self {
        Self {
            model_version: model_version.to_string(),
            calibration_version: calibration_version.to_string(),
        }
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new("jev-latest", "calib-canonical-v1")
    }
}

impl InferenceEngine for MockEngine {
    fn is_ready(&self) -> bool {
        true
    }

    fn model_version(&self) -> &str {
        &self.model_version
    }

    fn calibration_version(&self) -> &str {
        &self.calibration_version
    }

    fn decide(
        &self,
        state: &str,
        questions: &HashMap<String, Question>,
    ) -> Result<(HashMap<String, Answer>, UsageInfo), EngineError> {
        let mut answers = HashMap::with_capacity(questions.len());

        for (q_id, q) in questions {
            // Hash state and question to create deterministic pseudo-logits
            let text = format!("{}:{}:{}", state, q_id, q.text_content());
            let mut hasher_val: u64 = 14695981039346656037;
            for byte in text.bytes() {
                hasher_val ^= byte as u64;
                hasher_val = hasher_val.wrapping_mul(1099511628211);
            }

            match q {
                Question::Noul { .. } => {
                    let logit = ((hasher_val % 1000) as f32 / 100.0) - 5.0; // [-5.0, 5.0]
                    let prob = scaled_sigmoid(logit, 1.0);
                    answers.insert(
                        q_id.clone(),
                        Answer::Noul {
                            noul: (prob * 100.0).round() / 100.0,
                        },
                    );
                }
                Question::Choice { criteria, .. } => {
                    if criteria.is_empty() {
                        return Err(EngineError::InferenceFailed(format!(
                            "Question '{}' of type 'choice' requires non-empty criteria map",
                            q_id
                        )));
                    }

                    let mut opt_keys: Vec<String> = criteria.keys().cloned().collect();
                    opt_keys.sort(); // deterministic order

                    let mut logits = Vec::with_capacity(opt_keys.len());
                    for (opt_idx, opt_key) in opt_keys.iter().enumerate() {
                        let opt_str = format!("{}:{}", opt_key, opt_idx);
                        let mut h: u64 = hasher_val;
                        for b in opt_str.bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(1099511628211);
                        }
                        logits.push(((h % 1000) as f32 / 100.0) - 5.0);
                    }

                    let probs = scaled_softmax(&logits, 1.0);
                    let mut max_idx = 0;
                    let mut max_p = -1.0;
                    let mut dist = HashMap::new();

                    for (i, opt_key) in opt_keys.iter().enumerate() {
                        let p = (probs[i] * 10000.0).round() / 10000.0;
                        dist.insert(opt_key.clone(), p);
                        if probs[i] > max_p {
                            max_p = probs[i];
                            max_idx = i;
                        }
                    }

                    let chosen = opt_keys[max_idx].clone();
                    answers.insert(
                        q_id.clone(),
                        Answer::Choice {
                            choice: chosen,
                            confidence: (max_p * 100.0).round() / 100.0,
                            probabilities: dist,
                        },
                    );
                }
                Question::Score { criteria, .. } => {
                    if criteria.is_empty() {
                        return Err(EngineError::InferenceFailed(format!(
                            "Question '{}' of type 'score' requires non-empty criteria list",
                            q_id
                        )));
                    }

                    let num_levels = criteria.len();
                    let mut logits = Vec::with_capacity(num_levels);
                    for l in 0..num_levels {
                        let h: u64 = hasher_val ^ (l as u64);
                        logits.push(((h % 1000) as f32 / 100.0) - 5.0);
                    }

                    let probs = scaled_softmax(&logits, 1.0);
                    let mut expected_score = 0.0f32;
                    let mut legend = HashMap::new();
                    let mut dist = HashMap::new();
                    let mut max_p = 0.0f32;

                    for (i, &p) in probs.iter().enumerate() {
                        expected_score += (i as f32) * p;
                        legend.insert(i.to_string(), criteria[i].clone());
                        let rounded_p = (p * 10000.0).round() / 10000.0;
                        dist.insert(i.to_string(), rounded_p);
                        if p > max_p {
                            max_p = p;
                        }
                    }

                    answers.insert(
                        q_id.clone(),
                        Answer::Score {
                            score: (expected_score * 100.0).round() / 100.0,
                            confidence: (max_p * 100.0).round() / 100.0,
                            legend,
                            probabilities: dist,
                        },
                    );
                }
            }
        }

        let input_tokens = (state.len() / 4).max(1) + questions.len() * 15;
        let output_tokens = answers.len() * 20;
        let usage = UsageInfo {
            input_tokens,
            output_tokens,
        };

        Ok((answers, usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_engine_decide_all_primitives() {
        let engine = MockEngine::default();
        let state = "User asking for refund on damaged delivery.";

        let mut questions = HashMap::new();
        questions.insert(
            "is_urgent".to_string(),
            Question::Noul {
                instructions: "Urgent issue?".to_string(),
                criteria: None,
            },
        );

        let mut choice_criteria = HashMap::new();
        choice_criteria.insert("refund".to_string(), "Issue refund".to_string());
        choice_criteria.insert("replace".to_string(), "Replace item".to_string());
        questions.insert(
            "action".to_string(),
            Question::Choice {
                instructions: "What action?".to_string(),
                criteria: choice_criteria,
            },
        );

        questions.insert(
            "frustration".to_string(),
            Question::Score {
                instructions: "Level of frustration".to_string(),
                criteria: vec!["Low".to_string(), "Medium".to_string(), "High".to_string()],
            },
        );

        let (answers, usage) = engine
            .decide(state, &questions)
            .expect("Mock inference failed");
        assert_eq!(answers.len(), 3);
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);

        // Check Noul
        match answers.get("is_urgent").unwrap() {
            Answer::Noul { noul } => {
                assert!(*noul >= 0.0 && *noul <= 1.0);
            }
            _ => panic!("Expected Noul answer"),
        }

        // Check Choice
        match answers.get("action").unwrap() {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                assert!(choice == "refund" || choice == "replace");
                assert!(*confidence >= 0.0 && *confidence <= 1.0);
                assert_eq!(probabilities.len(), 2);
            }
            _ => panic!("Expected Choice answer"),
        }

        // Check Score
        match answers.get("frustration").unwrap() {
            Answer::Score {
                score,
                confidence,
                legend,
                probabilities,
            } => {
                assert!(*score >= 0.0 && *score <= 2.0);
                assert!(*confidence >= 0.0 && *confidence <= 1.0);
                assert_eq!(legend.len(), 3);
                assert_eq!(probabilities.len(), 3);
            }
            _ => panic!("Expected Score answer"),
        }
    }
}
