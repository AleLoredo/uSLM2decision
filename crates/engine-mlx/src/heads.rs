//! Decision heads evaluation in Rust: Noul, ChoiceHead, and ScoreHead.
//! Conforms to SPEC.md §3, ARCHITECTURE.md §3, and IMPLEMENTATION.md §4.

use crate::loader::HeadsWeights;
use openjev_calibration::temperature::{scaled_sigmoid, scaled_softmax};
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

    /// Evaluates Noul: boolean answer as integer 0 or 1
    pub fn evaluate_noul(&self, h_dec: &[f32]) -> u8 {
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
        if prob >= 0.5 {
            1
        } else {
            0
        }
    }

    /// Evaluates ChoiceHead:
    /// h_proj = W_choice * h_dec
    /// z_i = (e_{C_i}^T * h_proj) / sqrt(d)
    /// probs = scaled_softmax(z, T_choice)
    /// Returns (chosen_option_key, confidence, probabilities_map)
    pub fn evaluate_choice(
        &self,
        h_dec: &[f32],
        option_keys: &[String],
        option_embeddings: &[Vec<f32>],
    ) -> (String, f32, HashMap<String, f32>) {
        let d = self.weights.hidden_size;
        assert_eq!(h_dec.len(), d, "h_dec dimension mismatch");
        assert_eq!(
            option_keys.len(),
            option_embeddings.len(),
            "options and embeddings count mismatch"
        );

        // 1. Project h_dec: h_proj = W_choice * h_dec
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
        let mut logits = Vec::with_capacity(option_keys.len());
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

        for (idx, (opt_key, &prob)) in option_keys.iter().zip(probs.iter()).enumerate() {
            if prob > max_prob {
                max_prob = prob;
                best_idx = idx;
            }
            dist.insert(opt_key.clone(), (prob * 10000.0).round() / 10000.0);
        }

        let best_id = option_keys
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        let confidence = (max_prob * 10000.0).round() / 10000.0;
        (best_id, confidence, dist)
    }

    /// Evaluates ScoreHead:
    /// z_c = W_score[c] * h_dec + b_score[c]
    /// probs = scaled_softmax(z, T_score)
    /// Returns (expected_score, confidence, probabilities_map)
    pub fn evaluate_score(
        &self,
        h_dec: &[f32],
        num_levels: usize,
    ) -> (f32, f32, HashMap<String, f32>) {
        let d = self.weights.hidden_size;
        let max_classes = self.weights.score_classes;
        let k = num_levels.min(max_classes).max(2);
        assert_eq!(h_dec.len(), d, "h_dec dimension mismatch");

        // 1. Compute logits for levels
        let mut logits = Vec::with_capacity(k);
        for c in 0..k {
            let row_offset = c * d;
            let mut sum: f32 = 0.0;
            for (j, &h_val) in h_dec.iter().enumerate() {
                sum += self.weights.score_weight[row_offset + j] * h_val;
            }
            logits.push(sum + self.weights.score_bias[c]);
        }

        // 2. Softmax probabilities
        let probs = scaled_softmax(&logits, self.temperature_score);

        // 3. Expected score S = sum(probs * levels)
        let mut expected_score: f32 = 0.0;
        let mut max_prob = 0.0f32;
        let mut dist = HashMap::new();

        for (i, &p) in probs.iter().enumerate() {
            expected_score += (i as f32) * p;
            if p > max_prob {
                max_prob = p;
            }
            dist.insert(i.to_string(), (p * 10000.0).round() / 10000.0);
        }

        (
            (expected_score * 100.0).round() / 100.0,
            (max_prob * 10000.0).round() / 10000.0,
            dist,
        )
    }
}
