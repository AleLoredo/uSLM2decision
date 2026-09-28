//! Decision heads evaluation in Rust: BoolHead, ChoiceHead, and ScoreHead.
//! Conforms to SPEC.md §3, ARCHITECTURE.md §3, and IMPLEMENTATION.md §4.

use crate::loader::HeadsWeights;
use openjev_calibration::temperature::{scaled_sigmoid, scaled_softmax};
use openjev_core::types::ChoiceOption;
use std::collections::HashMap;

pub struct DecisionHeadsEvaluator<'a> {
    weights: &'a HeadsWeights,
    temperature_bool: f32,
    temperature_choice: f32,
    temperature_score: f32,
}

impl<'a> DecisionHeadsEvaluator<'a> {
    pub fn new(
        weights: &'a HeadsWeights,
        temperature_bool: f32,
        temperature_choice: f32,
        temperature_score: f32,
    ) -> Self {
        Self {
            weights,
            temperature_bool: if temperature_bool <= 0.0 {
                1.0
            } else {
                temperature_bool
            },
            temperature_choice: if temperature_choice <= 0.0 {
                1.0
            } else {
                temperature_choice
            },
            temperature_score: if temperature_score <= 0.0 {
                1.0
            } else {
                temperature_score
            },
        }
    }

    /// Evaluates BoolHead: z = w_bool^T * h_dec + b_bool
    /// Returns (value: bool, confidence: f32)
    pub fn evaluate_bool(&self, h_dec: &[f32]) -> (bool, f32) {
        assert_eq!(
            h_dec.len(),
            self.weights.hidden_size,
            "h_dec dimension mismatch"
        );

        let mut dot: f32 = 0.0;
        for (w, &h) in self.weights.bool_weight.iter().zip(h_dec.iter()) {
            dot += w * h;
        }
        let logit = dot + self.weights.bool_bias;
        let prob = scaled_sigmoid(logit, self.temperature_bool);
        let value = prob >= 0.5;
        let confidence = if value { prob } else { 1.0 - prob };
        (value, confidence)
    }

    /// Evaluates ChoiceHead:
    /// h_proj = W_choice * h_dec
    /// z_i = (e_{C_i}^T * h_proj) / sqrt(d)
    /// probs = scaled_softmax(z, T_choice)
    /// Returns (chosen_option_id, confidence, distribution_map)
    pub fn evaluate_choice(
        &self,
        h_dec: &[f32],
        options: &[ChoiceOption],
        option_embeddings: &[Vec<f32>],
    ) -> (String, f32, HashMap<String, f32>) {
        let d = self.weights.hidden_size;
        assert_eq!(h_dec.len(), d, "h_dec dimension mismatch");
        assert_eq!(
            options.len(),
            option_embeddings.len(),
            "options and embeddings count mismatch"
        );

        // 1. Project h_dec: h_proj = W_choice * h_dec
        // W_choice is stored in row-major [d, d]
        let mut h_proj = vec![0.0f32; d];
        for (row, h_out) in h_proj.iter_mut().enumerate() {
            let row_offset = row * d;
            let mut sum: f32 = 0.0;
            for (col, &h_val) in h_dec.iter().enumerate() {
                sum += self.weights.choice_weight[row_offset + col] * h_val;
            }
            *h_out = sum;
        }

        // 2. Bilinear dot-product for each option
        let scale = 1.0 / (d as f32).sqrt();
        let mut logits = Vec::with_capacity(options.len());
        for emb in option_embeddings {
            assert_eq!(emb.len(), d, "option embedding dimension mismatch");
            let mut dot: f32 = 0.0;
            for (&e, &p) in emb.iter().zip(h_proj.iter()) {
                dot += e * p;
            }
            logits.push(dot * scale);
        }

        // 3. Softmax
        let probs = scaled_softmax(&logits, self.temperature_choice);

        // 4. Find argmax
        let mut max_prob = -1.0f32;
        let mut best_idx = 0;
        let mut dist = HashMap::new();

        for (idx, (opt, &prob)) in options.iter().zip(probs.iter()).enumerate() {
            if prob > max_prob {
                max_prob = prob;
                best_idx = idx;
            }
            dist.insert(opt.id.clone(), prob);
        }

        let best_id = options
            .get(best_idx)
            .map(|o| o.id.clone())
            .unwrap_or_else(|| "default".to_string());

        (best_id, max_prob, dist)
    }

    /// Evaluates ScoreHead:
    /// z_c = W_score[c] * h_dec + b_score[c]
    /// probs = scaled_softmax(z, T_score)
    /// Returns (expected_score, discrete_level, confidence, variance, distribution_map)
    pub fn evaluate_score(
        &self,
        h_dec: &[f32],
        min_score: i32,
        max_score: i32,
    ) -> (f32, i32, f32, f32, Vec<f32>) {
        let d = self.weights.hidden_size;
        let num_levels = self.weights.score_classes;
        assert_eq!(h_dec.len(), d, "h_dec dimension mismatch");

        // 1. Compute logits for all ordinal classes
        let mut logits = Vec::with_capacity(num_levels);
        for c in 0..num_levels {
            let row_offset = c * d;
            let mut sum: f32 = 0.0;
            for (j, &h_val) in h_dec.iter().enumerate() {
                sum += self.weights.score_weight[row_offset + j] * h_val;
            }
            logits.push(sum + self.weights.score_bias[c]);
        }

        // 2. Softmax probabilities
        let probs = scaled_softmax(&logits, self.temperature_score);

        // 3. Ordinal levels array
        let min_f = min_score as f32;
        let max_f = max_score as f32;
        let step = if num_levels > 1 {
            (max_f - min_f) / ((num_levels - 1) as f32)
        } else {
            0.0
        };

        let mut levels = Vec::with_capacity(num_levels);
        for i in 0..num_levels {
            levels.push(min_f + (i as f32) * step);
        }

        // 4. Expected score S = sum(probs * levels)
        let mut expected_score: f32 = 0.0;
        for i in 0..num_levels {
            expected_score += probs[i] * levels[i];
        }

        // 5. Discrete level = round(expected_score) clamped to [min, max]
        let mut discrete_level = expected_score.round() as i32;
        if discrete_level < min_score {
            discrete_level = min_score;
        } else if discrete_level > max_score {
            discrete_level = max_score;
        }

        // 6. Variance = sum(probs * (levels - expected_score)^2)
        let mut variance: f32 = 0.0;
        for i in 0..num_levels {
            let diff = levels[i] - expected_score;
            variance += probs[i] * diff * diff;
        }

        // 7. Confidence = probability mass assigned to the discrete level
        let level_idx = if num_levels > 1 && (max_f - min_f) > 0.0 {
            let idx = ((discrete_level as f32 - min_f) / step).round() as usize;
            idx.min(num_levels - 1)
        } else {
            0
        };
        let confidence = probs[level_idx];

        (expected_score, discrete_level, confidence, variance, probs)
    }
}
