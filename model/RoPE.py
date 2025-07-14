import math
import torch
import torch.nn as nn

from model import DPS_Config

def precompute_rope_embeddings(config: DPS_Config) -> torch.Tensor:

    rope_dim = config.qk_rope_head_dim
    max_seq_len = config.max_seq_len
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    rope_theta = config.rope_theta
    rope_factor = config.rope_factor
    original_seq_len = config.original_seq_len
    device = config.device

    """
    Precomputes softened rotary positional embedding frequencies (complex exponentials).

    This version supports extrapolation smoothing like DeepSeek-V3.

    Returns:
        freqs_cis: [seq_len, rope_dim // 2], complex tensor of frequency rotations.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_idx, max_idx, dim):
        if min_idx == max_idx:
            max_idx += 1e-3  # prevent divide-by-zero
        ramp = (torch.arange(dim, dtype=torch.float32, device=device) - min_idx) / (max_idx - min_idx)
        return torch.clamp(ramp, 0, 1)

    assert rope_dim % 2 == 0, "Dimension must be divisible by 2"
    
    # creates the angles
    # creates the basic arange array then multiplies
    # then conducts the polar operation
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32)/ rope_dim))

    # Softening adjustment for long context
    if max_seq_len > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, rope_dim, rope_theta, original_seq_len)
        smooth = 1.0 - linear_ramp_factor(low, high, rope_dim // 2)
        inv_freq_soft = inv_freq / rope_factor
        inv_freq = inv_freq_soft * (1 - smooth) + inv_freq * smooth  # blend frequencies

    base_tensor = torch.arange(rope_dim, device=device).type_as(inv_freq)
    freqs = torch.outer(base_tensor, inv_freq)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def apply_rope_embeddings(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary embeddings using complex-valued multiplication.
    
    x: [batch, seq_len, rope_dim]
    freqs_cis: [seq_len, rope_dim // 2] — complex exponentials
    """
    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)