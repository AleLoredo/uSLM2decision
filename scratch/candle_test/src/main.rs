use candle_core::{Device, Tensor, DType};

fn main() -> anyhow::Result<()> {
    println!("=== Testing Candle on Apple Silicon Metal ===");
    let device = Device::new_metal(0)?;
    println!("Metal Device initialized: {:?}", device);

    // 1. Matrix multiplication test on Metal GPU
    let a = Tensor::randn(0.0f32, 1.0f32, (128, 512), &device)?;
    let b = Tensor::randn(0.0f32, 1.0f32, (512, 128), &device)?;
    let c = a.matmul(&b)?;
    println!("Matmul shape: {:?}", c.shape());

    // 2. Block-Causal Masking Test
    // State len S = 4, Query 1 len = 2, Query 2 len = 2. Total = 8
    let s = 4;
    let q1 = 2;
    let q2 = 2;
    let total = s + q1 + q2;

    let mut mask_data = vec![f32::NEG_INFINITY; total * total];
    // State attends to state (causal or full)
    for i in 0..s {
        for j in 0..=i { // Causal state
            mask_data[i * total + j] = 0.0;
        }
    }
    // Query 1 (positions 4..6) attends to all State (0..4) and to itself causally (4..=i)
    for i in s..(s + q1) {
        for j in 0..s { // full state
            mask_data[i * total + j] = 0.0;
        }
        for j in s..=i { // causal within Q1
            mask_data[i * total + j] = 0.0;
        }
    }
    // Query 2 (positions 6..8) attends to all State (0..4) and to itself causally (6..=i)
    for i in (s + q1)..total {
        for j in 0..s { // full state
            mask_data[i * total + j] = 0.0;
        }
        for j in (s + q1)..=i { // causal within Q2
            mask_data[i * total + j] = 0.0;
        }
    }

    let mask_tensor = Tensor::from_vec(mask_data, (1, 1, total, total), &device)?;
    println!("Block-Causal Mask Tensor created on Metal: {:?}", mask_tensor.shape());

    // 3. Simulated Attention Scores + Mask Additive
    let qk_scores = Tensor::zeros((1, 8, total, total), DType::F32, &device)?; // 8 heads
    let masked_scores = qk_scores.broadcast_add(&mask_tensor)?;
    let attn_probs = candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;
    println!("Attn Probs computed on Metal: shape {:?}", attn_probs.shape());

    // 4. Decision Heads Projections
    let hidden_dim = 512;
    let h_q1 = Tensor::randn(0.0f32, 1.0f32, (1, hidden_dim), &device)?;
    
    // Bool Head: sigma(W_bool * h_Q + b)
    let w_bool = Tensor::randn(0.0f32, 1.0f32, (hidden_dim, 1), &device)?;
    let bool_logit = h_q1.matmul(&w_bool)?;
    let p_bool = candle_nn::ops::sigmoid(&bool_logit)?;
    println!("Bool Head P(true): {:?}", p_bool.to_vec2::<f32>()?[0][0]);

    // Choice Head (corrected formula): z_i = (h_Q^T W_choice e_Ci) / sqrt(d)
    let num_options = 4;
    let w_choice = Tensor::randn(0.0f32, 1.0f32, (hidden_dim, hidden_dim), &device)?;
    let e_options = Tensor::randn(0.0f32, 1.0f32, (num_options, hidden_dim), &device)?;
    
    let h_proj = h_q1.matmul(&w_choice)?; // (1, d)
    let z = (h_proj.matmul(&e_options.t()?)? / (hidden_dim as f64).sqrt())?; // (1, N)
    let choice_dist = candle_nn::ops::softmax(&z, candle_core::D::Minus1)?;
    println!("Choice Head Distribution (N=4): {:?}", choice_dist.to_vec2::<f32>()?[0]);

    // Score Head (ordinal expected value & variance)
    let num_levels = 5;
    let w_score = Tensor::randn(0.0f32, 1.0f32, (hidden_dim, num_levels), &device)?;
    let score_logits = h_q1.matmul(&w_score)?;
    let score_probs = candle_nn::ops::softmax(&score_logits, candle_core::D::Minus1)?;
    let probs_vec = score_probs.to_vec2::<f32>()?[0].clone();
    
    let levels: Vec<f32> = (0..num_levels).map(|i| i as f32).collect();
    let expected_score: f32 = probs_vec.iter().zip(levels.iter()).map(|(p, l)| p * l).sum();
    let variance: f32 = probs_vec.iter().zip(levels.iter()).map(|(p, l)| p * (l - expected_score).powi(2)).sum();
    println!("Score Head: expected_score = {:.3}, discrete_level = {}, variance = {:.3}", expected_score, expected_score.round() as i32, variance);

    println!("\n>>> ALL CHECKS PASSED ON METAL GPU <<<");
    Ok(())
}
