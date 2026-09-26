//! Block-causal attention masking and position-ID generation per ARCHITECTURE.md §2.

pub const MASK_VALUE: f32 = -1e9;

/// Generates a single-request additive attention mask [Total, Total].
/// 0.0 = attend, -1e9 = masked out.
/// Layout: [State (s_len)] [Query_1 (q_lens[0])] ... [Query_K (q_lens[K-1])]
pub fn create_block_causal_mask(s_len: usize, q_lens: &[usize], state_causal: bool) -> Vec<f32> {
    let total: usize = s_len + q_lens.iter().sum::<usize>();
    let mut mask = vec![MASK_VALUE; total * total];

    // 1. State tokens
    for i in 0..s_len {
        let max_j = if state_causal { i + 1 } else { s_len };
        for j in 0..max_j {
            mask[i * total + j] = 0.0;
        }
    }

    // 2. Query tokens
    let mut current_q_offset = s_len;
    for &q_len in q_lens {
        let q_start = current_q_offset;
        let q_end = current_q_offset + q_len;

        for i in q_start..q_end {
            // Query attends to ALL state tokens
            for j in 0..s_len {
                mask[i * total + j] = 0.0;
            }
            // Query attends to itself causally
            for j in q_start..=i {
                mask[i * total + j] = 0.0;
            }
        }
        current_q_offset += q_len;
    }

    mask
}

/// Constructs per-segment position IDs:
/// State: 0 .. S-1
/// Each Query_k: S .. S + Q_k - 1 (resets to S for every query segment).
pub fn create_segment_position_ids(s_len: usize, q_lens: &[usize]) -> Vec<i32> {
    let mut pos_ids = Vec::with_capacity(s_len + q_lens.iter().sum::<usize>());

    // State positions: 0 .. S-1
    for i in 0..s_len {
        pos_ids.push(i as i32);
    }

    // Query positions: reset to S for each query
    for &q_len in q_lens {
        for i in 0..q_len {
            pos_ids.push((s_len + i) as i32);
        }
    }

    pos_ids
}

/// Packs multiple requests into a single unified sequence with complete
/// mathematical isolation across requests (off-diagonal blocks set to -1e9).
pub fn pack_multi_request_masks(
    request_lengths: &[(usize, Vec<usize>)],
    state_causal: bool,
) -> (Vec<f32>, Vec<i32>, usize) {
    let mut req_totals = Vec::new();
    let mut total_tokens = 0usize;

    for (s_len, q_lens) in request_lengths {
        let req_total = s_len + q_lens.iter().sum::<usize>();
        req_totals.push(req_total);
        total_tokens += req_total;
    }

    let mut global_mask = vec![MASK_VALUE; total_tokens * total_tokens];
    let mut global_pos_ids = Vec::with_capacity(total_tokens);

    let mut req_offset = 0usize;
    for (s_len, q_lens) in request_lengths {
        let single_mask = create_block_causal_mask(*s_len, q_lens, state_causal);
        let single_pos = create_segment_position_ids(*s_len, q_lens);
        let req_total = s_len + q_lens.iter().sum::<usize>();

        // Copy intra-request block into global mask
        for i in 0..req_total {
            for j in 0..req_total {
                global_mask[(req_offset + i) * total_tokens + (req_offset + j)] =
                    single_mask[i * req_total + j];
            }
        }

        // Append position IDs
        global_pos_ids.extend_from_slice(&single_pos);
        req_offset += req_total;
    }

    (global_mask, global_pos_ids, total_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_request_block_causal_mask() {
        let s_len = 3;
        let q_lens = [2, 2];
        let total = 7;
        let mask = create_block_causal_mask(s_len, &q_lens, true);

        assert_eq!(mask.len(), total * total);

        // State causal: token 0 attends to 0; token 1 to 0..=1; token 2 to 0..=2
        assert_eq!(mask[0 * total + 0], 0.0);
        assert_eq!(mask[0 * total + 1], MASK_VALUE);
        assert_eq!(mask[1 * total + 0], 0.0);
        assert_eq!(mask[1 * total + 1], 0.0);
        assert_eq!(mask[1 * total + 2], MASK_VALUE);

        // Query 1 (pos 3..5) attends to all state (0..3)
        assert_eq!(mask[3 * total + 0], 0.0);
        assert_eq!(mask[3 * total + 1], 0.0);
        assert_eq!(mask[3 * total + 2], 0.0);
        assert_eq!(mask[3 * total + 3], 0.0); // causal self
        assert_eq!(mask[3 * total + 4], MASK_VALUE); // future self
        assert_eq!(mask[3 * total + 5], MASK_VALUE); // query 2 (isolated!)

        // Query 2 (pos 5..7) attends to state (0..3) but NOT Query 1 (3..5)
        assert_eq!(mask[5 * total + 0], 0.0);
        assert_eq!(mask[5 * total + 2], 0.0);
        assert_eq!(mask[5 * total + 3], MASK_VALUE); // cannot attend to Query 1
        assert_eq!(mask[5 * total + 4], MASK_VALUE); // cannot attend to Query 1
        assert_eq!(mask[5 * total + 5], 0.0); // causal self
    }

    #[test]
    fn test_segment_position_ids() {
        let s_len = 3;
        let q_lens = [2, 3];
        let pos_ids = create_segment_position_ids(s_len, &q_lens);

        // State: 0, 1, 2
        // Q1: 3, 4
        // Q2: 3, 4, 5 (resets to S=3!)
        assert_eq!(pos_ids, vec![0, 1, 2, 3, 4, 3, 4, 5]);
    }

    #[test]
    fn test_multi_request_packing_isolation() {
        let reqs = vec![(2, vec![2]), (2, vec![2])];
        let (mask, pos_ids, total) = pack_multi_request_masks(&reqs, true);

        assert_eq!(total, 8); // 4 + 4
        assert_eq!(pos_ids, vec![0, 1, 2, 3, 0, 1, 2, 3]);

        // Tokens in req 1 (0..4) looking at req 2 (4..8) MUST be MASK_VALUE
        for i in 0..4 {
            for j in 4..8 {
                assert_eq!(mask[i * total + j], MASK_VALUE);
            }
        }

        // Tokens in req 2 (4..8) looking at req 1 (0..4) MUST be MASK_VALUE
        for i in 4..8 {
            for j in 0..4 {
                assert_eq!(mask[i * total + j], MASK_VALUE);
            }
        }
    }
}
