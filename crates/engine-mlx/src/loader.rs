//! Safetensors and architecture loader for OpenJEV decision models.
//! Conforms to SPEC.md §2, ARCHITECTURE.md §3, and IMPLEMENTATION.md §4.

use crate::types::EngineError;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use tracing::info;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelArchitectureConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

fn default_hidden_size() -> usize {
    896
}

fn default_num_layers() -> usize {
    24
}

fn default_num_heads() -> usize {
    14
}

fn default_vocab_size() -> usize {
    151936
}

impl Default for ModelArchitectureConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_layers(),
            num_attention_heads: default_num_heads(),
            vocab_size: default_vocab_size(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeadsWeights {
    pub hidden_size: usize,
    pub bool_weight: Vec<f32>, // [hidden_size]
    pub bool_bias: f32,
    pub choice_weight: Vec<f32>, // [hidden_size, hidden_size]
    pub score_weight: Vec<f32>,  // [num_classes, hidden_size]
    pub score_bias: Vec<f32>,    // [num_classes]
    pub score_classes: usize,
}

impl HeadsWeights {
    /// Loads and strictly validates canonical decision heads from a safetensors file.
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        expected_hidden_size: usize,
    ) -> Result<Self, EngineError> {
        let p = path.as_ref();
        if !p.exists() {
            return Err(EngineError::InferenceFailed(format!(
                "Heads safetensors file not found at: {}",
                p.display()
            )));
        }

        let bytes = fs::read(p).map_err(|e| {
            EngineError::InferenceFailed(format!("Failed to read {}: {}", p.display(), e))
        })?;

        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| {
            EngineError::InferenceFailed(format!("Safetensors deserialization error: {}", e))
        })?;

        // 1. Bool head
        let bool_w_view = tensors.tensor("bool_head.linear.weight").map_err(|_| {
            EngineError::InferenceFailed("Missing 'bool_head.linear.weight'".into())
        })?;
        let bool_w = read_f32_tensor(&bool_w_view)?;
        if bool_w.len() != expected_hidden_size {
            return Err(EngineError::InferenceFailed(format!(
                "bool_head.linear.weight size mismatch: expected {}, got {}",
                expected_hidden_size,
                bool_w.len()
            )));
        }

        let bool_b_view = tensors
            .tensor("bool_head.linear.bias")
            .map_err(|_| EngineError::InferenceFailed("Missing 'bool_head.linear.bias'".into()))?;
        let bool_b = read_f32_tensor(&bool_b_view)?;
        let bool_bias = bool_b.first().copied().unwrap_or(0.0);

        // 2. Choice head
        let choice_w_view = tensors.tensor("choice_head.w_choice.weight").map_err(|_| {
            EngineError::InferenceFailed("Missing 'choice_head.w_choice.weight'".into())
        })?;
        let choice_w = read_f32_tensor(&choice_w_view)?;
        let expected_choice_len = expected_hidden_size * expected_hidden_size;
        if choice_w.len() != expected_choice_len {
            return Err(EngineError::InferenceFailed(format!(
                "choice_head.w_choice.weight size mismatch: expected {}x{}={}, got {}",
                expected_hidden_size,
                expected_hidden_size,
                expected_choice_len,
                choice_w.len()
            )));
        }

        // 3. Score head
        let score_w_view = tensors.tensor("score_head.w_score.weight").map_err(|_| {
            EngineError::InferenceFailed("Missing 'score_head.w_score.weight'".into())
        })?;
        let score_w = read_f32_tensor(&score_w_view)?;
        let score_classes = score_w.len() / expected_hidden_size;
        if score_w.len() % expected_hidden_size != 0 || score_classes == 0 {
            return Err(EngineError::InferenceFailed(format!(
                "score_head.w_score.weight size invalid for hidden_size {}: got {}",
                expected_hidden_size,
                score_w.len()
            )));
        }

        let score_b_view = tensors.tensor("score_head.w_score.bias").map_err(|_| {
            EngineError::InferenceFailed("Missing 'score_head.w_score.bias'".into())
        })?;
        let score_b = read_f32_tensor(&score_b_view)?;
        if score_b.len() != score_classes {
            return Err(EngineError::InferenceFailed(format!(
                "score_head.w_score.bias size mismatch: expected {}, got {}",
                score_classes,
                score_b.len()
            )));
        }

        info!(
            "Loaded canonical heads: hidden_size={}, score_classes={}",
            expected_hidden_size, score_classes
        );

        Ok(Self {
            hidden_size: expected_hidden_size,
            bool_weight: bool_w,
            bool_bias,
            choice_weight: choice_w,
            score_weight: score_w,
            score_bias: score_b,
            score_classes,
        })
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_f32_tensor(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, EngineError> {
    let data = view.data();
    if data.len() % 4 != 0 {
        return Err(EngineError::InferenceFailed(
            "Invalid f32 byte alignment".into(),
        ));
    }
    let f32_count = data.len() / 4;
    let mut vec = Vec::with_capacity(f32_count);
    for chunk in data.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vec.push(val);
    }
    Ok(vec)
}

pub fn load_architecture_config<P: AsRef<Path>>(path: P) -> ModelArchitectureConfig {
    let p = path.as_ref();
    if p.exists() {
        if let Ok(content) = fs::read_to_string(p) {
            if let Ok(cfg) = serde_json::from_str::<ModelArchitectureConfig>(&content) {
                return cfg;
            }
        }
    }
    ModelArchitectureConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_heads_safetensors_if_present() {
        let path = Path::new("../../models/qwen2.5-0.5b/heads.safetensors");
        if path.exists() {
            let heads =
                HeadsWeights::load_from_file(path, 896).expect("Failed to load canonical heads");
            assert_eq!(heads.hidden_size, 896);
            assert_eq!(heads.bool_weight.len(), 896);
            assert_eq!(heads.choice_weight.len(), 896 * 896);
            assert_eq!(heads.score_classes, 5);
        }
    }
}
