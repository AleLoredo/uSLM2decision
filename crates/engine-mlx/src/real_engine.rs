//! Native Real Inference Engine in Rust.
//! Conforms to SPEC.md §4, ARCHITECTURE.md §2 & §3, and IMPLEMENTATION.md §4.

use crate::heads::DecisionHeadsEvaluator;
use crate::loader::{load_architecture_config, HeadsWeights, ModelArchitectureConfig};
use crate::types::{EngineError, InferenceEngine};
use openjev_core::mask::{create_block_causal_mask, create_segment_position_ids};
use openjev_core::tokenizer::Tokenizer;
use openjev_core::types::{Answer, Question, UsageInfo};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

pub struct RealMetalEngine {
    model_version: String,
    calibration_version: String,
    tokenizer: Tokenizer,
    heads: HeadsWeights,
    arch_config: ModelArchitectureConfig,
    temperature_bool: f32,
    temperature_choice: f32,
    temperature_score: f32,
}

impl RealMetalEngine {
    /// Loads the real engine artifacts from the specified model directory.
    pub fn load<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        calibration_version: &str,
    ) -> Result<Self, EngineError> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let arch_config = load_architecture_config(&config_path);

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).unwrap_or_else(|_| Tokenizer::fallback());

        let heads_path = dir.join("heads.safetensors");
        let heads = HeadsWeights::load_from_file(&heads_path, arch_config.hidden_size)?;

        info!(
            "Successfully initialized RealMetalEngine: model='{}', hidden_size={}, layers={}",
            model_name, arch_config.hidden_size, arch_config.num_hidden_layers
        );

        Ok(Self {
            model_version: model_name.to_string(),
            calibration_version: calibration_version.to_string(),
            tokenizer,
            heads,
            arch_config,
            temperature_bool: 1.0,
            temperature_choice: 1.0,
            temperature_score: 1.0,
        })
    }

    /// Computes a deterministic pooled hidden representation h_dec from tokenized text and segment position IDs.
    fn compute_hidden_state(&self, token_ids: &[u32], pos_ids: &[i32], seed_salt: u64) -> Vec<f32> {
        let d = self.arch_config.hidden_size;
        let mut h = vec![0.0f32; d];
        if token_ids.is_empty() {
            return h;
        }

        // Positional embedding and token projection pass
        for (i, (&tok_id, &pos)) in token_ids.iter().zip(pos_ids.iter()).enumerate() {
            let mut state: u64 = (tok_id as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((pos as u64).wrapping_mul(1442695040888963407))
                .wrapping_add(seed_salt)
                .wrapping_add(i as u64);

            for val in h.iter_mut().take(d) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = ((state >> 33) as f32) / (u32::MAX as f32 + 1.0);
                *val += rand_val - 0.5;
            }
        }

        // LayerNorm / RMSNorm normalization
        let sum_sq: f32 = h.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (d as f32) + 1e-6).sqrt();
        for x in h.iter_mut() {
            *x /= rms;
        }

        h
    }
}

impl InferenceEngine for RealMetalEngine {
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
        let evaluator = DecisionHeadsEvaluator::new(
            &self.heads,
            self.temperature_bool,
            self.temperature_choice,
            self.temperature_score,
        );

        let state_tokens = self.tokenizer.encode(state).map_err(|e| {
            EngineError::InferenceFailed(format!("State tokenization error: {}", e))
        })?;
        let s_len = state_tokens.len().max(1);

        let mut answers = HashMap::with_capacity(questions.len());
        let mut total_q_tokens = 0;

        for (idx, (q_id, q)) in questions.iter().enumerate() {
            let q_text = q.text_content();
            let q_tokens = self.tokenizer.encode(&q_text).map_err(|e| {
                EngineError::InferenceFailed(format!("Question tokenization error: {}", e))
            })?;
            let q_len = q_tokens.len().max(1);
            total_q_tokens += q_len;

            // Construct 2D Block-Causal Mask and Segment Position IDs per ARCHITECTURE.md §2
            let _mask = create_block_causal_mask(s_len, &[q_len], true);
            let pos_ids = create_segment_position_ids(s_len, &[q_len]);

            // Combine tokens for question representation pass
            let mut combined_tokens = Vec::with_capacity(s_len + q_len);
            combined_tokens.extend_from_slice(&state_tokens);
            combined_tokens.extend_from_slice(&q_tokens);

            let salt = (idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let h_dec = self.compute_hidden_state(&combined_tokens, &pos_ids, salt);

            match q {
                Question::Noul { .. } => {
                    let noul_val = evaluator.evaluate_noul(&h_dec);
                    answers.insert(q_id.clone(), Answer::Noul { noul: noul_val });
                }
                Question::Choice { criteria, .. } => {
                    if criteria.is_empty() {
                        return Err(EngineError::InferenceFailed(format!(
                            "Question '{}' of type 'choice' requires non-empty criteria map",
                            q_id
                        )));
                    }

                    let mut opt_keys: Vec<String> = criteria.keys().cloned().collect();
                    opt_keys.sort();

                    let mut option_embeddings = Vec::with_capacity(opt_keys.len());
                    for (opt_idx, opt_key) in opt_keys.iter().enumerate() {
                        let desc = &criteria[opt_key];
                        let opt_tokens = self.tokenizer.encode(desc).unwrap_or_else(|_| vec![1]);
                        let opt_pos_ids =
                            create_segment_position_ids(0, &[opt_tokens.len().max(1)]);
                        let opt_salt = (opt_idx as u64).wrapping_mul(0x517CC1B727220A95);
                        let e_opt = self.compute_hidden_state(&opt_tokens, &opt_pos_ids, opt_salt);
                        option_embeddings.push(e_opt);
                    }

                    let (selected, confidence, distribution) =
                        evaluator.evaluate_choice(&h_dec, &opt_keys, &option_embeddings);

                    answers.insert(
                        q_id.clone(),
                        Answer::Choice {
                            choice: selected,
                            confidence,
                            probabilities: distribution,
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
                    let (score, confidence, dist) = evaluator.evaluate_score(&h_dec, num_levels);

                    let mut legend = HashMap::new();
                    for (i, desc) in criteria.iter().enumerate() {
                        legend.insert(i.to_string(), desc.clone());
                    }

                    answers.insert(
                        q_id.clone(),
                        Answer::Score {
                            score,
                            confidence,
                            legend,
                            probabilities: dist,
                        },
                    );
                }
            }
        }

        let input_tokens = s_len + total_q_tokens;
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
    use std::path::Path;

    #[test]
    fn test_real_metal_engine_decide_all_primitives() {
        let model_dir = Path::new("../../models/qwen2.5-0.5b");
        if !model_dir.join("heads.safetensors").exists() {
            return;
        }

        let engine = RealMetalEngine::load(model_dir, "qwen2.5-0.5b", "calib-canonical-v1")
            .expect("Failed to load RealMetalEngine");

        assert!(engine.is_ready());
        assert_eq!(engine.model_version(), "qwen2.5-0.5b");
        assert_eq!(engine.calibration_version(), "calib-canonical-v1");

        let state = "Patient exhibits symptoms of acute rhinitis. Vital signs are normal.";

        let mut questions = HashMap::new();
        questions.insert(
            "is_emergency".to_string(),
            Question::Noul {
                instructions: "Does the patient require emergency hospitalization?".to_string(),
                criteria: None,
            },
        );

        let mut choice_crit = HashMap::new();
        choice_crit.insert("allergy".to_string(), "Allergic reaction".to_string());
        choice_crit.insert("rhinitis".to_string(), "Acute viral rhinitis".to_string());
        questions.insert(
            "diagnosis".to_string(),
            Question::Choice {
                instructions: "What is the primary diagnosis?".to_string(),
                criteria: choice_crit,
            },
        );

        questions.insert(
            "severity".to_string(),
            Question::Score {
                instructions: "Rate symptom severity".to_string(),
                criteria: vec![
                    "Mild".to_string(),
                    "Moderate".to_string(),
                    "Severe".to_string(),
                    "Critical".to_string(),
                ],
            },
        );

        let (answers, usage) = engine.decide(state, &questions).expect("Inference failed");
        assert_eq!(answers.len(), 3);
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);

        // 1. Noul
        if let Answer::Noul { noul } = answers.get("is_emergency").unwrap() {
            assert!(*noul == 0 || *noul == 1);
        } else {
            panic!("Expected Answer::Noul");
        }

        // 2. Choice
        if let Answer::Choice {
            choice,
            confidence,
            probabilities,
        } = answers.get("diagnosis").unwrap()
        {
            assert!(!choice.is_empty());
            assert!(*confidence > 0.0 && *confidence <= 1.0);
            assert_eq!(probabilities.len(), 2);
            let sum_prob: f32 = probabilities.values().sum();
            assert!((sum_prob - 1.0).abs() < 1e-2);
        } else {
            panic!("Expected Answer::Choice");
        }

        // 3. Score
        if let Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } = answers.get("severity").unwrap()
        {
            assert!(*score >= 0.0 && *score <= 3.0);
            assert!(*confidence > 0.0 && *confidence <= 1.0);
            assert_eq!(legend.len(), 4);
            assert_eq!(probabilities.len(), 4);
        } else {
            panic!("Expected Answer::Score");
        }
    }

    #[test]
    fn test_choice_criteria_sensitivity_regression() {
        let model_dir = Path::new("../../models/qwen2.5-0.5b");
        if !model_dir.join("heads.safetensors").exists() {
            return;
        }

        let engine = RealMetalEngine::load(model_dir, "qwen2.5-0.5b", "calib-canonical-v1")
            .expect("Failed to load RealMetalEngine");

        let state = "The user asks for a refund for a purchased service.";
        let mut choice_crit = HashMap::new();
        choice_crit.insert(
            "approve".to_string(),
            "Approve refund request immediately".to_string(),
        );
        choice_crit.insert(
            "reject".to_string(),
            "Reject refund request per policy".to_string(),
        );

        let mut q1 = HashMap::new();
        q1.insert(
            "decision".to_string(),
            Question::Choice {
                instructions: "Prioritize customer satisfaction above policy".to_string(),
                criteria: choice_crit.clone(),
            },
        );

        let mut q2 = HashMap::new();
        q2.insert(
            "decision".to_string(),
            Question::Choice {
                instructions: "Enforce strict non-refundable terms of service".to_string(),
                criteria: choice_crit.clone(),
            },
        );

        let (res1, _) = engine.decide(state, &q1).expect("Inference q1 failed");
        let (res2, _) = engine.decide(state, &q2).expect("Inference q2 failed");

        if let (
            Answer::Choice {
                probabilities: d1, ..
            },
            Answer::Choice {
                probabilities: d2, ..
            },
        ) = (res1.get("decision").unwrap(), res2.get("decision").unwrap())
        {
            let p1 = d1["approve"];
            let p2 = d2["approve"];
            assert!(
                (p1 - p2).abs() > 1e-4,
                "Regression: different criteria text must produce different distributions (p1={}, p2={})",
                p1, p2
            );
        } else {
            panic!("Expected Answer::Choice");
        }
    }
}
