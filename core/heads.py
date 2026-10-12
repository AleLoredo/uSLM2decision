"""
OpenJEV Core Decision Heads Module (MLX Backend)
Implements typed heads per ARCHITECTURE.md §3 and SPEC.md §3:
1. BoolHead: P(true) = sigmoid(W_bool * h_Q + b), confidence in [0.5, 1.0]
2. ChoiceHead: Corrected formula z_i = (h_Q^T W_choice e_{C_i}) / sqrt(d)
3. ScoreHead: Softmax distribution, expected score S_bar, discrete level, variance
"""

import math
from typing import Dict, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn

class BoolHead(nn.Module):
    """
    Bool Head (SPEC.md §3.1 & ARCHITECTURE.md §3.1)
    Calculates P(true) = sigmoid(W_bool * h_Q + b).
    Confidence is reported relative to chosen boolean value in range [0.5, 1.0].
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def __call__(self, h_q: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Args:
            h_q: Query final token pooled representations, shape (B, d)
        Returns:
            values: Boolean decision tensor (B,)
            confidences: Confidence values in range [0.5, 1.0], shape (B,)
            p_true: Raw P(true) probabilities, shape (B,)
        """
        logits = self.linear(h_q).squeeze(-1) # (B,)
        p_true = mx.sigmoid(logits)
        
        # Tie-breaking at exactly 0.5 defaults to False
        values = p_true > 0.5
        confidences = mx.where(values, p_true, 1.0 - p_true)
        
        return values, confidences, p_true

class ChoiceHead(nn.Module):
    """
    Choice Head — Corrected Formula with L2 Normalization (ARCHITECTURE.md §3.2 & §6.2)
    Scores query representation h_Q against each option embedding e_{C_i}:
    z_i = ( (h_Q^T W_choice / ||...||) (e_{C_i} / ||...||) ) / tau
    P(C_i) = softmax(z)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_choice = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(
        self,
        h_q: mx.array,
        e_options: mx.array,
        temperature: float = 1.0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Args:
            h_q: Query pooled representation, shape (B, d)
            e_options: Option embeddings, shape (B, N, d) or (N, d)
            temperature: Softmax scaling temperature T > 0
        Returns:
            selected_indices: Index of chosen option for each batch, shape (B,)
            confidences: Prob mass of selected option, shape (B,)
            probs: Normalized probability distribution over N options, shape (B, N)
        """
        if e_options.ndim == 2:
            # Expand options (N, d) to batch (B, N, d)
            e_options = mx.expand_dims(e_options, 0)
            e_options = mx.broadcast_to(e_options, (h_q.shape[0], e_options.shape[1], e_options.shape[2]))

        # Residual projection: preserves pretrained semantic alignment while allowing learned refinement
        h_proj = h_q + self.w_choice(h_q)

        # L2-normalize projected query and option vectors to prevent representation anisotropy blowup
        h_norm = h_proj / (mx.linalg.norm(h_proj, axis=-1, keepdims=True) + 1e-6)
        h_norm = mx.expand_dims(h_norm, 1) # (B, 1, d)

        e_norm = e_options / (mx.linalg.norm(e_options, axis=-1, keepdims=True) + 1e-6) # (B, N, d)

        # Cosine dot product: (B, 1, d) x (B, d, N) -> (B, 1, N) -> (B, N)
        cos_sim = mx.squeeze(mx.matmul(h_norm, mx.transpose(e_norm, (0, 2, 1))), axis=1) # (B, N)

        # Scale by contrastive temperature (tau = 0.03 * temperature)
        tau = 0.03 * max(temperature, 1e-4)
        z = cos_sim / tau

        probs = mx.softmax(z, axis=-1)
        selected_indices = mx.argmax(probs, axis=-1)
        confidences = mx.max(probs, axis=-1)

        return selected_indices, confidences, probs

class ScoreHead(nn.Module):
    """
    Score Head — Ordinal-aware (SPEC.md §3.3 & ARCHITECTURE.md §3.3 & §6.3)
    Computes distribution over M discrete score levels, expected score, discrete level, and variance.
    """
    def __init__(self, hidden_dim: int, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.w_score = nn.Linear(hidden_dim, num_levels)

    def __call__(
        self,
        h_q: mx.array,
        min_val: int = 0,
        max_val: Optional[int] = None,
        num_levels: Optional[int] = None
    ) -> Dict[str, mx.array]:
        """
        Args:
            h_q: Query pooled representation, shape (B, d)
            min_val: Minimum score level integer (default: 0)
            max_val: Maximum score level integer
            num_levels: Explicit count of levels in rubric
        Returns:
            Dict containing expected score, discrete level, confidence, variance, and distribution
        """
        if num_levels is None:
            if max_val is not None:
                num_levels = max_val - min_val + 1
            else:
                num_levels = self.num_levels

        # Clamp num_levels to at most the linear layer dimension
        k = min(num_levels, self.num_levels)

        logits = self.w_score(h_q)[:, :k] # (B, k)
        probs = mx.softmax(logits, axis=-1) # (B, k)

        if max_val is None:
            max_val = min_val + k - 1

        levels = mx.linspace(min_val, max_val, k) # (k,)

        # Expected score S_bar = sum(i * p_i)
        expected_score = mx.sum(probs * levels, axis=-1) # (B,)

        # Discrete level = round(S_bar) clamped to [min, max]
        discrete_level = mx.clip(mx.round(expected_score), min_val, max_val).astype(mx.int32)

        # Variance sigma_S^2 = sum((l - S_bar)^2 * p_i)
        levels_exp = mx.expand_dims(levels, 0) # (1, k)
        exp_score_exp = mx.expand_dims(expected_score, -1) # (B, 1)
        diff_sq = (levels_exp - exp_score_exp) ** 2 # (B, k)
        variance = mx.sum(probs * diff_sq, axis=-1) # (B,)

        # Confidence is probability mass assigned to the discrete_level
        level_indices = mx.clip(discrete_level - min_val, 0, k - 1)
        batch_indices = mx.arange(h_q.shape[0])
        confidence = probs[batch_indices, level_indices]

        return {
            "score": expected_score,
            "discrete_level": discrete_level,
            "confidence": confidence,
            "variance": variance,
            "distribution": probs,
            "logits": logits
        }

class OptionEncoder(nn.Module):
    """
    Option Encoder (ARCHITECTURE.md §3.2 resolved decision)
    Computes option embeddings e_{C_i} as a dedicated forward-encoding pass reusing
    the base embedding & encoder layers over each option's description.
    """
    def __init__(self, embed_layer: nn.Module, encoder_layers: list, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.embed = embed_layer
        self.encoder = encoder_layers
        self.norm = norm_layer

    def __call__(self, option_token_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            option_token_ids: Option description token IDs, shape (N, L)
        Returns:
            e_options: Pooled option embeddings, shape (N, d)
        """
        x = self.embed(option_token_ids) # (N, L, d)
        for layer in self.encoder:
            x = layer(x, mask=attention_mask)
        if self.norm is not None:
            x = self.norm(x)
        # Mean pool non-padding tokens
        if attention_mask is not None:
            mask_expanded = mx.expand_dims(attention_mask, -1).astype(mx.float32)
            e_options = mx.sum(x * mask_expanded, axis=1) / mx.clip(mx.sum(mask_expanded, axis=1), 1e-9, None)
        else:
            e_options = mx.mean(x, axis=1)
        return e_options
