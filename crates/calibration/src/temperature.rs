//! Post-hoc calibration functions: temperature scaling and calibration metrics.

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn scale_logit(logit: f32, temperature: f32) -> f32 {
    let t = if temperature <= 0.0 { 1.0 } else { temperature };
    logit / t
}

pub fn scaled_sigmoid(logit: f32, temperature: f32) -> f32 {
    sigmoid(scale_logit(logit, temperature))
}

pub fn scaled_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = if temperature <= 0.0 { 1.0 } else { temperature };
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / t).exp())
        .collect();
    let sum_exp: f32 = exp_values.iter().sum();

    if sum_exp == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp_values.iter().map(|&v| v / sum_exp).collect()
    }
}

pub fn compute_ece(confidences: &[f32], accuracies: &[bool], num_bins: usize) -> f32 {
    if confidences.is_empty() || confidences.len() != accuracies.len() {
        return 0.0;
    }

    let bin_size = 1.0 / (num_bins as f32);
    let mut bin_counts = vec![0usize; num_bins];
    let mut bin_conf_sums = vec![0.0f32; num_bins];
    let mut bin_acc_sums = vec![0.0f32; num_bins];

    for (&conf, &acc) in confidences.iter().zip(accuracies.iter()) {
        let mut bin_idx = (conf / bin_size).floor() as usize;
        if bin_idx >= num_bins {
            bin_idx = num_bins - 1;
        }
        bin_counts[bin_idx] += 1;
        bin_conf_sums[bin_idx] += conf;
        bin_acc_sums[bin_idx] += if acc { 1.0 } else { 0.0 };
    }

    let total = confidences.len() as f32;
    let mut ece = 0.0f32;

    for i in 0..num_bins {
        if bin_counts[i] > 0 {
            let avg_conf = bin_conf_sums[i] / (bin_counts[i] as f32);
            let avg_acc = bin_acc_sums[i] / (bin_counts[i] as f32);
            let weight = (bin_counts[i] as f32) / total;
            ece += weight * (avg_conf - avg_acc).abs();
        }
    }

    ece
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = scaled_softmax(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_perfect_calibration_ece_is_zero() {
        let confidences = vec![0.8, 0.8, 0.8, 0.8, 0.8];
        let accuracies = vec![true, true, true, true, false]; // 4/5 = 0.8
        let ece = compute_ece(&confidences, &accuracies, 10);
        assert!(ece < 0.05);
    }
}
