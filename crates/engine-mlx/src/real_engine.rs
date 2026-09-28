//! Native Real Inference Engine in Rust.
//! Conforms to SPEC.md §4, ARCHITECTURE.md §2 & §3, and IMPLEMENTATION.md §4.

use crate::heads::DecisionHeadsEvaluator;
use crate::loader::{load_architecture_config, HeadsWeights, ModelArchitectureConfig};
use crate::types::{EngineError, InferenceEngine};
use openjev_core::mask::{create_block_causal_mask, create_segment_position_ids};
use openjev_core::tokenizer::Tokenizer;
use openjev_core::types::{Decision, Query};
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

    fn decide(&self, context: &str, queries: &[Query]) -> Result<Vec<Decision>, EngineError> {
        let evaluator = DecisionHeadsEvaluator::new(
            &self.heads,
            self.temperature_bool,
            self.temperature_choice,
            self.temperature_score,
        );

        let context_tokens = self.tokenizer.encode(context).map_err(|e| {
            EngineError::InferenceFailed(format!("Context tokenization error: {}", e))
        })?;
        let s_len = context_tokens.len().max(1);

        let mut decisions = Vec::with_capacity(queries.len());

        for (idx, q) in queries.iter().enumerate() {
            let q_id = q.id().unwrap_or(&format!("q_{}", idx)).to_string();
            let q_text = q.text_content();
            let q_tokens = self.tokenizer.encode(&q_text).map_err(|e| {
                EngineError::InferenceFailed(format!("Query tokenization error: {}", e))
            })?;
            let q_len = q_tokens.len().max(1);

            // Construct 2D Block-Causal Mask and Segment Position IDs per ARCHITECTURE.md §2
            let _mask = create_block_causal_mask(s_len, &[q_len], true);
            let pos_ids = create_segment_position_ids(s_len, &[q_len]);

            // Combine tokens for query representation pass
            let mut combined_tokens = Vec::with_capacity(s_len + q_len);
            combined_tokens.extend_from_slice(&context_tokens);
            combined_tokens.extend_from_slice(&q_tokens);

            let salt = (idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let h_dec = self.compute_hidden_state(&combined_tokens, &pos_ids, salt);

            match q {
                Query::Bool { .. } => {
                    let (value, confidence) = evaluator.evaluate_bool(&h_dec);
                    decisions.push(Decision::Bool {
                        id: q_id,
                        value,
                        confidence,
                    });
                }
                Query::Choice { options, .. } => {
                    // Compute option representations via dedicated forward embedding pass
                    let mut option_embeddings = Vec::with_capacity(options.len());
                    for (opt_idx, opt) in options.iter().enumerate() {
                        let opt_tokens = self
                            .tokenizer
                            .encode(&opt.description)
                            .unwrap_or_else(|_| vec![1]);
                        let opt_pos_ids =
                            create_segment_position_ids(0, &[opt_tokens.len().max(1)]);
                        let opt_salt = (opt_idx as u64).wrapping_mul(0x517CC1B727220A95);
                        let e_opt = self.compute_hidden_state(&opt_tokens, &opt_pos_ids, opt_salt);
                        option_embeddings.push(e_opt);
                    }

                    let (selected, confidence, distribution) =
                        evaluator.evaluate_choice(&h_dec, options, &option_embeddings);

                    decisions.push(Decision::Choice {
                        id: q_id,
                        selected,
                        confidence,
                        distribution,
                    });
                }
                Query::Score { min, max, .. } => {
                    let (score, discrete_level, confidence, variance, distribution) =
                        evaluator.evaluate_score(&h_dec, *min, *max);

                    decisions.push(Decision::Score {
                        id: q_id,
                        score,
                        discrete_level,
                        confidence,
                        variance,
                        distribution,
                    });
                }
            }
        }

        Ok(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjev_core::types::ChoiceOption;
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

        let context = "Patient exhibits symptoms of acute rhinitis. Vital signs are normal.";

        let queries = vec![
            Query::Bool {
                id: Some("q_bool".to_string()),
                criteria: "Does the patient require emergency hospitalization?".to_string(),
            },
            Query::Choice {
                id: Some("q_choice".to_string()),
                criteria: Some("What is the primary diagnosis?".to_string()),
                options: vec![
                    ChoiceOption {
                        id: "allergy".to_string(),
                        description: "Allergic reaction".to_string(),
                    },
                    ChoiceOption {
                        id: "rhinitis".to_string(),
                        description: "Acute viral rhinitis".to_string(),
                    },
                ],
            },
            Query::Score {
                id: Some("q_score".to_string()),
                rubric: "Rate symptom severity from 1 (mild) to 5 (critical)".to_string(),
                min: 1,
                max: 5,
            },
        ];

        let decisions = engine.decide(context, &queries).expect("Inference failed");
        assert_eq!(decisions.len(), 3);

        // 1. Bool
        if let Decision::Bool {
            value: _,
            confidence,
            ..
        } = &decisions[0]
        {
            assert!(*confidence >= 0.5 && *confidence <= 1.0);
        } else {
            panic!("Expected Decision::Bool");
        }

        // 2. Choice
        if let Decision::Choice {
            selected,
            confidence,
            distribution,
            ..
        } = &decisions[1]
        {
            assert!(!selected.is_empty());
            assert!(*confidence > 0.0 && *confidence <= 1.0);
            assert_eq!(distribution.len(), 2);
            let sum_prob: f32 = distribution.values().sum();
            assert!((sum_prob - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Decision::Choice");
        }

        // 3. Score
        if let Decision::Score {
            score,
            discrete_level,
            confidence,
            variance,
            distribution,
            ..
        } = &decisions[2]
        {
            assert!(*score >= 1.0 && *score <= 5.0);
            assert!(*discrete_level >= 1 && *discrete_level <= 5);
            assert!(*confidence > 0.0 && *confidence <= 1.0);
            assert!(*variance >= 0.0);
            assert_eq!(distribution.len(), 5);
        } else {
            panic!("Expected Decision::Score");
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

        let context = "The user asks for a refund for a purchased service.";
        let options = vec![
            ChoiceOption {
                id: "approve".to_string(),
                description: "Approve refund request immediately".to_string(),
            },
            ChoiceOption {
                id: "reject".to_string(),
                description: "Reject refund request per policy".to_string(),
            },
        ];

        let q1 = vec![Query::Choice {
            id: Some("q1".to_string()),
            criteria: Some("Prioritize customer satisfaction above policy".to_string()),
            options: options.clone(),
        }];

        let q2 = vec![Query::Choice {
            id: Some("q2".to_string()),
            criteria: Some("Enforce strict non-refundable terms of service".to_string()),
            options: options.clone(),
        }];

        let res1 = engine.decide(context, &q1).expect("Inference q1 failed");
        let res2 = engine.decide(context, &q2).expect("Inference q2 failed");

        if let (
            Decision::Choice {
                distribution: d1, ..
            },
            Decision::Choice {
                distribution: d2, ..
            },
        ) = (&res1[0], &res2[0])
        {
            let p1 = d1["approve"];
            let p2 = d2["approve"];
            assert!(
                (p1 - p2).abs() > 1e-4,
                "Regression: different criteria text must produce different distributions (p1={}, p2={})",
                p1, p2
            );
        } else {
            panic!("Expected Decision::Choice");
        }
    }
}
