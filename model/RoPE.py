import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from model import DPS_Config

def precompute_rope_embeddings(config: DPS_Config):
    # creates the angles
    # creates the basic arange array then multiplies
    # then conducts the polar operation
    inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, config.rope_dim, 2, dtype=torch.float32)/ config.rope_dim))

    # if sequence length is more than the original sequence length
    

    base_tensor = torch.arange(config.rope_dim, device=config.device).type_as(inv_freq)
    freqs = torch.outer(base_tensor, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs).conj()

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