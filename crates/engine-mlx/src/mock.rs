//! Deterministic Calibrated Mock Engine for testing and CI.

use crate::types::{EngineError, InferenceEngine};
use openjev_calibration::temperature::{scaled_sigmoid, scaled_softmax};
use openjev_core::types::{Decision, Query};
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
        Self::new("qwen2.5-0.5b-mock", "calib-canonical-v1")
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

    fn decide(&self, context: &str, queries: &[Query]) -> Result<Vec<Decision>, EngineError> {
        let mut decisions = Vec::with_capacity(queries.len());

        for (idx, q) in queries.iter().enumerate() {
            let q_id = q.id().unwrap_or(&format!("q_{}", idx)).to_string();

            // Hash context and query text to create a deterministic score
            let text = format!("{}:{}:{}", context, q.text_content(), idx);
            let mut hasher_val: u64 = 14695981039346656037;
            for byte in text.bytes() {
                hasher_val ^= byte as u64;
                hasher_val = hasher_val.wrapping_mul(1099511628211);
            }

            match q {
                Query::Bool { .. } => {
                    let logit = ((hasher_val % 1000) as f32 / 100.0) - 5.0; // [-5.0, 5.0]
                    let prob = scaled_sigmoid(logit, 1.2);
                    let val = prob >= 0.5;
                    let conf = if val { prob } else { 1.0 - prob };

                    decisions.push(Decision::Bool {
                        id: q_id,
                        value: val,
                        confidence: conf,
                    });
                }
                Query::Choice { options, .. } => {
                    if options.is_empty() {
                        return Err(EngineError::InferenceFailed(format!(
                            "Query '{}' of type 'choice' requires options",
                            q_id
                        )));
                    }

                    let mut logits = Vec::with_capacity(options.len());
                    for (opt_idx, opt) in options.iter().enumerate() {
                        let opt_str = format!("{}:{}", opt.id, opt_idx);
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

                    for (i, opt) in options.iter().enumerate() {
                        let p = probs[i];
                        dist.insert(opt.id.clone(), p);
                        if p > max_p {
                            max_p = p;
                            max_idx = i;
                        }
                    }

                    decisions.push(Decision::Choice {
                        id: q_id,
                        selected: options[max_idx].id.clone(),
                        confidence: max_p,
                        distribution: dist,
                    });
                }
                Query::Score { min, max, .. } => {
                    let min_f = *min as f32;
                    let max_f = *max as f32;
                    let num_levels = (*max - *min + 1).max(2) as usize;

                    let mut logits = Vec::with_capacity(num_levels);
                    for l in 0..num_levels {
                        let h: u64 = hasher_val ^ (l as u64);
                        logits.push(((h % 1000) as f32 / 100.0) - 5.0);
                    }

                    let dist = scaled_softmax(&logits, 1.0);
                    let mut expected_score = 0.0f32;
                    for (i, &p) in dist.iter().enumerate() {
                        expected_score += (min_f + i as f32) * p;
                    }

                    let mut variance = 0.0f32;
                    for (i, &p) in dist.iter().enumerate() {
                        let val = min_f + i as f32;
                        variance += p * (val - expected_score).powi(2);
                    }

                    let discrete = expected_score.round().clamp(min_f, max_f) as i32;
                    let conf = (1.0 / (1.0 + variance)).clamp(0.0, 1.0);

                    decisions.push(Decision::Score {
                        id: q_id,
                        score: expected_score,
                        discrete_level: discrete,
                        confidence: conf,
                        variance,
                        distribution: dist,
                    });
                }
            }
        }

        Ok(decisions)
    }
}
