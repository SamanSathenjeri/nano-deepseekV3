import math
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F # type: ignore
from dataclasses import dataclass
from typing import Optional
from utils import load_config

@dataclass
class DPS_Config:
    """
    Defines model hyperparameters and arguments from config.yaml file

    Attributes:
        num_embd(int) = number of dimensions to represent the token
        num_layers(int) = number of transformer blocks
        num_dense_layers(int) = First N layers which are dense
        num_attention_heads(int) = Number of attention heads per block (head_dimension = num_embd//num_attention heads)
        intermediate_size(int) = Overall FeedForward dim size (4 * n_embd)

        # MLA
        latent_dim(int) = Dimension of the compressed latent space (head dim//2 or 4)
        proj_matrix_size(int) = the proj matrices to up and down project
        q_lora_rank(int) = LoRA rank for query projections.
        kv_lora_rank(int) = LoRA rank for key-value projections.
        qk_nope_head_dim(int) = Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim(int) = Dimension for query-key projections with rotary embeddings.
        v_head_dim(int) = Dimension for value projections.

        # MoE
        num_experts(int) = Total number of routed experts per MoE layer
        num_shared_experts(int) = Number of experts that get activated no matter the token
        num_activated_experts(int) = K: num of experts activated per token
        expert_inter_size(int) = Intermediate size for each expert's FFN (intermediate_size/num_experts * 2)
        n_expert_groups(int) = Number of groups experts are grouped in
        n_limited_groups(int) = Number of experts per group
        route_scale(int) = Scaling factor per route
        score_function(string) = The scoring function used to determine what expert/s to route to in MOE

        # RoPE
        original_seq_len(int) = Original sequence length
        rope_theta(int) = Base for rotary positional encoding
        rope_factor(int) = Scaling factor for extended sequence lengths
        beta_fast(int) = Fast beta correction factor
        beta_slow(int) = Slow beta correction factor
        mscale(int) = Scaling factor for extended attention

        # training
        batch_size(int) = Maximum number of batches trained at the same time
        num_epochs(int) = number of complete passes through the dataset
        learning_rate(int) = learning rate for training
        warmup_steps(int) = starting steps with low learning rate
        num_predicted_tokens(int) = number of future tokens that are simultaneously predicted
        max_seq_len(int) = Maximum sequence length.

        # data
        block_size(int) = number of tokens in the vocabulary
        vocab_size(int) = Context size
        dtype(string) = Data type to represent floats

        # device
        device(string) = Device to be used for training
        world_size(int) = number of processes to distribute across
        
    """

    config = load_config()

    num_embd = config['model']['num_embd']
    num_layers = config['model']['num_layers']
    num_dense_layers = config['model']['num_dense_layers']
    num_attention_heads = config['model']['num_attention_heads']
    intermediate_size = config['model']['intermediate_size']

    # MLA
    latent_dim = config['model']['latent_dim']
    proj_matrix_size = config['model']['proj_matrix_size']
    q_lora_rank = config['model']['q_lora_rank']
    kv_lora_rank = config['model']['kv_lora_rank']
    qk_nope_head_dim = config['model']['qk_nope_head_dim']
    qk_rope_head_dim = config['model']['qk_rope_head_dim']
    v_head_dim = config['model']['v_head_dim']

    # MoE
    num_experts = config['model']['num_experts']
    num_shared_experts = config['model']['num_shared_experts']
    num_activated_experts = config['model']['num_activated_experts']
    expert_inter_size = config['model']['expert_inter_size']
    n_expert_groups = config['model']['n_expert_groups']
    n_limited_groups = config['model']['n_limited_groups']
    route_scale = config['model']['route_scale']
    score_function = config['model']['score_function']

    # RoPE
    original_seq_len = config['model']['original_seq_len']
    rope_theta = config['model']['rope_theta']
    rope_factor = config['model']['rope_factor']
    beta_fast = config['model']['beta_fast']
    beta_slow = config['model']['beta_slow']
    mscale = config['model']['mscale']

    # training
    max_batch_size = config['training']['max_batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    num_predicted_tokens = config['training']['num_predicted_tokens']
    max_seq_len = config['training']['max_seq_len']

    # data
    # block_size = config['data']['max_seq_len']
    vocab_size = config['data']['vocab_size']
    dtype = config['data']['dtype']

    # device
    device = config['device']['device']
    world_size = config['device']['world_size']

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
            max_idx = max_idx + 1e-3  # prevent divide-by-zero
        ramp = (torch.arange(dim, dtype=torch.float32, device=device) - min_idx) / (max_idx - min_idx)
        return torch.clamp(ramp, 0, 1)

    assert rope_dim % 2 == 0, "Dimension must be divisible by 2"
    
    # creates the angles
    # creates the basic arange array then multiplies
    # then conducts the polar operation
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))

    # Softening adjustment for long context
    if max_seq_len > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, rope_dim, rope_theta, original_seq_len)
        smooth = 1.0 - linear_ramp_factor(low, high, rope_dim // 2)
        inv_freq_soft = inv_freq / rope_factor
        inv_freq = inv_freq_soft * (1 - smooth) + inv_freq * smooth  # blend frequencies

    base_tensor = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype) # Use max_seq_len for positions
    freqs = torch.outer(base_tensor, inv_freq)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def apply_rope_embeddings(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, start_pos=0):
    """
    Applies RoPE embeddings to query and key vectors.
    
    q, k: [B, T, H, D]
    freqs_cis: [seq_len, D//2] (complex)
    """
    q_dtype = q.dtype
    k_dtype = k.dtype
    B, T, H, D = q.shape
    assert D % 2 == 0, f"Last dim {D} must be even to form complex pairs"
    D_half = D // 2

    q = q.to(torch.float32)
    k = k.to(torch.float32)

    # Convert to complex
    q_complex = torch.view_as_complex(q.view(B, T, H, D_half, 2))
    k_complex = torch.view_as_complex(k.view(B, T, H, D_half, 2))

    # Slice precomputed RoPE frequencies
    freqs_slice = freqs_cis[start_pos : start_pos + T]  # [T, D_half]
    freqs_slice = freqs_slice.view(1, T, 1, D_half)     # broadcastable

    # Apply rotation
    q_rotated = q_complex * freqs_slice
    k_rotated = k_complex * freqs_slice

    q_out = torch.view_as_real(q_rotated).view(B, T, H, D)
    k_out = torch.view_as_real(k_rotated).view(B, T, H, D)

    return q_out.to(q_dtype), k_out.to(k_dtype)

class Embedding(nn.Module):
    """
    The embedding layer

    Attributes:
        weight (nn.Parameter): The embedding matrix
    """
    def __init__(self, vocab_size: int, num_embd: int):
        """
        Initializing the embedding layer

        Args:
            vocab_size (int): Vocabulary size.
            num_embd (int): Number of dimensions to represent tokens
        """
        super().__init__()
        # creates a weight parameter that holds the assigned part of the dictionary
        self.weight = nn.Parameter(torch.empty(vocab_size, num_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass of the embedding layer

        Args:
            x (torch.Tensor): input tokens

        Returns: 
            y (torch.Tensor): (input, num_embd) size tensor of token embeddings
        """
        # the translation of tokens to embeddings for this processor
        y = F.embedding(x, self.weight)
        return y

class RMSNorm(nn.Module):
    """
    Defines the RMSNorm component of the Transformer

    Attributes:
        num_embd(int): Number of dimensions to represent tokens
        eps(float): a variable used to avoid division by 0 and massive inflation by a tiny sum of squares
        weight(nn.Parameter): normalized matrix
    """
    def __init__(self, num_embd: int, eps: float = 1e-6): 
        '''
        Initializes the RMSNorm component

        Arguments:
            num_embd(int): Number of dimensions to represent tokens
            eps(float): a variable used to avoid division by 0 and massive inflation by a tiny sum of squares
        '''
        super().__init__()
        self.num_embd = num_embd
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_embd))

    def forward(self, x: torch.Tensor):
        '''
        Forward Pass of the RMSNorm component

        Arguments:
            x(torch.Tensor):

        Returns:
            torch.Tensor:
        '''
        return F.rms_norm(x, (self.num_embd,), self.weight, self.eps)
    
class MLP(nn.Module):
    '''
    Defines the MLP component of the FFN 

    Attributes:
        layer1(nn.Linear): Linear layer that up-projects from num_embd to intermediate_size (to be used in silu)
        layer2(nn.Linear): Linear layer to down-project from intermediate_size to num_embd
        layer3(nn.Linear): Linear layer that acts as gating mechanism for layer 1 - also projects from num_embd to intermediate_size
    '''

    def __init__(self, num_embd: int, intermediate_size: int):
        '''
        Initializes the MLP component

        Attributes:
            num_embd(int): Number of dimensions to represent tokens
            intermediate_size(int): Intermediate dimension 

        !!!more about the SWiGLU gating mechanism!!!
        the output from silu(layer1(x)) is a nonlinear activation 
        the output from layer3(x) is the raw output from the linear layer's transformations
        they are element-wise multiplied where the layer3(x)'s output acts as a "gate" for the output from silu(layer1(x))
        what happens is if the activations from layer3(x) is high, then it will amplify the corresponding feature from the silu
        else, if the activation from the layer3(x) is low, then it will diminish and "close the gate" of the corresponding 
        feature from the silu - this allows us highlight or hide certain features based on what our input is
        '''
        super().__init__()
        self.layer1 = nn.Linear(num_embd, intermediate_size) # Sets up linear layer to use SiLU
        self.layer2 = nn.Linear(intermediate_size, num_embd) # Sets up linear layer to project back to num_embd dimensions
        self.layer3 = nn.Linear(num_embd, intermediate_size) # Sets up linear layer to be used in SWiGLU gating mechanism

    def forward(self, x: torch.Tensor):
        '''
        Forward Pass of the MLP layer using SWiGLU
        
        Arguments:
            x(torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: output tensor after MLP operations
        '''
        return self.layer2(F.silu(self.layer1(x)) * self.layer3(x))

class Router(nn.Module):
    def __init__(self, num_embd, num_experts, top_k=2):
        """
        Router for Mixture of Experts
        num_embd: hidden dimension of the input
        num_experts: total number of experts
        top_k: number of experts to select per token
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(num_embd, num_experts, bias=False)

    def forward(self, x):
        """
        Input: x -> [batch, seq_len, dim]
        Output: dispatch_mask, combine_weights, selected_experts
        """
        # Compute logits for each expert
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        # Softmax over experts to get normalized gate scores
        gate_scores = F.softmax(logits, dim=-1)

        # Select top-k experts per token
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # [batch, seq_len, top_k]

        # Create one-hot dispatch mask
        dispatch_mask = torch.zeros_like(gate_scores).unsqueeze(-1)  # [batch, seq_len, num_experts, 1]
        batch_indices = torch.arange(x.size(0), device=x.device)[:, None, None]
        seq_indices = torch.arange(x.size(1), device=x.device)[None, :, None]

        for k in range(self.top_k):
            dispatch_mask[batch_indices, seq_indices, topk_indices[..., k], 0] = 1.0

        # Normalize combine weights
        combine_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)  # [batch, seq_len, top_k]

        return dispatch_mask, combine_weights, topk_indices


class MOE(nn.Module):
    def __init__(self, dim, num_experts, top_k=2, hidden_dim=2048):
        """
        Mixture of Experts module
        dim: input and output dimension
        num_experts: number of experts
        top_k: number of experts per token
        hidden_dim: dimension of the expert's FFN layer
        """
        super().__init__()
        self.router = Router(dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Input: x -> [batch, seq_len, dim]
        Output: y -> [batch, seq_len, dim]
        """
        dispatch_mask, combine_weights, topk_indices = self.router(x)

        expert_outputs = torch.zeros_like(x)
        for k in range(self.router.top_k):
            selected_expert_idx = topk_indices[..., k]  # [batch, seq_len]
            mask = (selected_expert_idx.unsqueeze(-1) == torch.arange(self.router.num_experts, device=x.device)).any(dim=-1)

            for expert_idx in range(self.router.num_experts):
                expert_mask = (selected_expert_idx == expert_idx)  # [batch, seq_len]
                if expert_mask.any():
                    tokens = x[expert_mask]  # [num_tokens, dim]
                    processed = self.experts[expert_idx](tokens)  # FFN
                    expert_outputs[expert_mask] += processed * combine_weights[..., k][expert_mask].unsqueeze(-1)

        return expert_outputs

class MLA(nn.Module):
    '''
    The multiheaded latent attention module will allow the tokens to share context to each other

    Attributes:
        dim (int): The dimensionality of the input and output embeddings (num_embd)
        n_heads (int): The number of attention heads
        latent_dim (int): The dimensionality of the latent space for attention computation
        head_dim (int): The dimensionality of a single attention head
        q_proj (nn.Linear): Linear layer for projecting input to queries
        k_proj (nn.Linear): Linear layer for projecting input to keys
        v_proj (nn.Linear): Linear layer for projecting input to values
        out_proj (nn.Linear): Linear layer for projecting the final attention output
        compress_q (nn.Linear): Linear layer to compress query heads to the latent dimension
        compress_k (nn.Linear): Linear layer to compress key heads to the latent dimension
        compress_v (nn.Linear): Linear layer to compress value heads to the latent dimension
        decompress (nn.Linear): Linear layer to decompress the latent output back to the head dimension
        softmax_scale (float): The scaling factor for the softmax function, calculated as 1.0/ 
        latent_dim
    '''
    def __init__(self, config):
        '''
        Initialization of the MLA component

        Arguments:
            config(DPS_Config): holds the model's hyperparameters
        '''
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.latent_dim = config.latent_dim
        self.head_dim = config.num_embd // self.num_attention_heads
        assert config.num_embd % self.num_attention_heads == 0

        self.ln = nn.LayerNorm(config.num_embd)

        self.q_proj = nn.Linear(config.num_embd, config.num_embd)
        self.k_proj = nn.Linear(config.num_embd, config.num_embd)
        self.v_proj = nn.Linear(config.num_embd, config.num_embd)
        self.out_proj = nn.Linear(config.num_embd, config.num_embd)

        self.compress_q = nn.Linear(self.head_dim, self.latent_dim)
        self.compress_k = nn.Linear(self.head_dim, self.latent_dim)
        self.compress_v = nn.Linear(self.head_dim, self.latent_dim)
        self.decompress = nn.Linear(self.latent_dim, self.head_dim)

        self.attn_dropout = nn.Dropout(getattr(config, "attn_dropout", 0.0))
        self.out_dropout = nn.Dropout(getattr(config, "out_dropout", 0.0))

        self.softmax_scale = 1.0 / math.sqrt(self.latent_dim)

    def forward(self, x: torch.Tensor, start_pos=0, freqs_cis=None, mask=None):
        '''
        Forward

        Arguments:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dim)
            start_pos (int, optional): The starting position of the sequence, used for slicing rotary embeddings. Defaults to 0
            freqs_cis (torch.Tensor, optional): Precomputed rotary embeddings
            mask (torch.Tensor, optional): A causal mask tensor of shape (sequence_length, sequence_length) to prevent attention to future tokens

        Returns:
            attn_output (torch.Tensor): The output tensor after the multi-head latent attention mechanism, with the same shape as the input x
        '''
        B, T, D = x.size()
        H = self.num_attention_heads
        Hd = self.head_dim
        L = self.latent_dim
        x_norm = self.ln(x)

        q = self.q_proj(x_norm).view(B, T, H, Hd)
        k = self.k_proj(x_norm).view(B, T, H, Hd)
        v = self.v_proj(x_norm).view(B, T, H, Hd)

        if freqs_cis is not None:
            freqs_cis = freqs_cis[start_pos:start_pos+T, :]
        q, k = apply_rope_embeddings(q, k, freqs_cis)  # NOTE ****** ensure apply_rope accepts (B,T,H,Hd)

        # compress to latent
        # result shapes: (B, T, H, L)
        q_lat = self.compress_q(q)
        k_lat = self.compress_k(k)
        v_lat = self.compress_v(v)

        # merge batch and heads for fast matmul
        # (B*H, T, L)
        q_lat = q_lat.permute(0,2,1,3).contiguous().view(B*H, T, L)
        k_lat = k_lat.permute(0,2,1,3).contiguous().view(B*H, T, L)
        v_lat = v_lat.permute(0,2,1,3).contiguous().view(B*H, T, L)

        # (B*H, T, T)
        scores = torch.matmul(q_lat, k_lat.transpose(-2, -1)) * self.softmax_scale

        # mask: allow bool or additive mask
        if mask is not None:
            # mask expected shape (T,T) or (B,T,T). Convert to additive -inf mask broadcastable to (B*H,T,T)
            if mask.dtype == torch.bool:
                additive = (~mask).to(scores.dtype) * -1e9  # (T,T) or (B,T,T)
            else:
                additive = mask  # assume already additive
            # expand to (B*H, T, T)
            if additive.dim() == 2:
                additive = additive.unsqueeze(0)  # (1,T,T)
            additive = additive.expand(B, -1, -1).reshape(B*H, T, T)
            scores = scores + additive

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # attn @ v -> (B*H, T, L)
        out = torch.matmul(attn, v_lat)

        # unmerge heads: (B, H, T, L) -> (B, T, H, L)
        out = out.view(B, H, T, L).permute(0,2,1,3).contiguous()

        # decompress per-head latent -> head_dim: (B,T,H,Hd)
        out = self.decompress(out)

        out = out.view(B, T, D)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out
        
class Block(nn.Module):
    """
    Defines Transformer Block architecture

    Attributes:
        attn(nn.Module): Attention layer using Multi-head Latent Attention (MLA)
        ffn(nn.Module): FFN using either MLP (for dense layers) or MoE (to use experts)
        attn_norm(nn.Module): Layer normalization before the attention
        ffn_norm(nn.Module): Layer normalization before the ffn
    """

    def __init__(self, layer_index: int, config: DPS_Config):
        """
        Initializes the transformer block

        Arguments:
            layer_index(int): layer index in the transformer (used to see if dense layer or MoE layer)
            config(Config): holds model hyperparameter information
        """
        super().__init__()
        self.attn = MLA(config)
        self.ffn = MLP(config.num_embd, config.intermediate_size) if layer_index < config.num_dense_layers else MOE(config)
        self.attn_norm = RMSNorm(config.num_embd)
        self.ffn_norm = RMSNorm(config.num_embd)

    def forward(self, x: torch.Tensor, start_pos: int, rotary_embeddings: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Transformer block

        Arguments:
            x(torch.Tensor): Input tensor of tokens
            start_pos(int): Starting position of the sequence
            rotary_embeddings(torch.Tensor): exponential values for rotary embeddings
            mask(Optional[torch.Tensor]): mask tensor to hide future tokens from influencing attention

        Return:
            x(torch.Tensor): Outputted residual tensor after undergoing operations
        """
        x = x + self.attn(self.attn_norm(x), start_pos, rotary_embeddings, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class DPS(nn.Module):

    """
        Creates transformer model with RoPE embeddings, layers, and output projection

        Attributes:
            config(Config): holds model hyperparameter information
            transformer(nn.ModuleDict): dictionary of transformer architecture pieces
            embed(Embedding): The embedding layer for input tokens up projection
            layers(nn.ModuleList): holds transformer blocks (attention + ffn + norm)
            rms_norm(RMSNorm): last layer of layer normalization after all the blocks have been computed
            output_proj(OutputProjection): the output projection layer to down project to output
            rotary_embeddings(torch.Tensor): exponential values for rotary embeddings
    """

    def __init__(self, config: DPS_Config):

        """
        Initializes the Transformer Model

        Arguments:
            config (Config): holds model's hyperparameters
        """

        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embed = Embedding(config.vocab_size, config.num_embd), # creates the embedding (up projection)
            layers = nn.ModuleList([Block(i, config) for i in range(config.num_layers)]), # creates num_layers of transformer blocks
            rms_norm = RMSNorm(config.num_embd), # adds in an RMS norm at the end, after all of the transformer blocks
            output_proj = nn.Linear(config.num_embd, config.vocab_size) # finally down projects to get final predictions
        ))

        self.register_buffer("rotary_embeddings", precompute_rope_embeddings(config), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):

        """
        Forward pass of the Transformer

        Arguments:
            tokens (torch.Tensor): input tensor of tokens with the shape of (batch_size, sequence_length)
            start_pos(int): the starting position of the sequence

        Returns:
            logits(torch.Tensor): Logits tensor of shape (batch_size, vocab_size)
        """

        # creates embedded tokens and rotary embeddings
        sequence_length = tokens.size(1)
        embedded_tokens = self.transformer.embed(tokens)
        rotary_embeddings = self.rotary_embeddings[start_pos : start_pos + sequence_length]
        mask = None

        # creates mask of upper triangle with -inf elements, if sequence length is above the trivial amount of 1
        # to avoid the model from using future tokens to help it calculate the attention (the triu_ is an upper triangle mask)
        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float("-inf"), device=tokens.device).triu(1)
        
        # takes in the output of every layer
        for layer in self.transformer.layers:
            embedded_tokens = layer(embedded_tokens, start_pos, rotary_embeddings, mask)
        
        # normalizes the tokens, slices to get the predicted tokens only, and down projects to logits
        embedded_tokens = self.transformer.rms_norm(embedded_tokens)[:, -1]
        logits = self.transformer.output_proj(embedded_tokens)

        return logits        
    
if __name__ == "__main__":
    config = DPS_Config() # creates a new model Config
    # torch.set_default_dtype(torch.bfloat16) #sets the datatype to Bfloat 16 (shortened mantissa)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.device)
    torch.manual_seed(0)
    x = torch.randint(0, config.vocab_size, (2, 128)) # creates a sample tensor of random values with size (2, 128)
    model = DPS(config) # creates a model using the config
    print(model(x).size())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)