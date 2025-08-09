# Nano DeepSeek-V3: A Tiny Yet Powerful Transformer Model

A faithful representation, miniaturized, implementation of DeepSeek-V3 architecture, designed for research exploration and efficient experimentation on limited hardware (T4 GPU). Trained on the FineWebEDU dataset to enrich the model with high quality data.

<!-- INSERT GIF HERE **************************
https://github.com/charmbracelet/vhs -->

---

### 🚀 Features
This project is a from-scratch, minimal-yet-faithful reproduction of DeepSeek-V3, incorporating advanced architectural components such as:
- 🔁 Rotary Positional Embeddings (RoPE)
- 🔧 RMSNorm
- 🧠 Multi-Latent Attention
- 🧩 MoE-style (Mixture of Experts) Blocks
- ⚡ LoRA integration for efficient fine-tuning

| Component              | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| Rotary Embeds          | Injects position using RoPE, allowing for extrapolation and better performance  |
| LoRA                   | Low-Rank Adaptation for efficient finetuning                                    |
| RMSNorm                | RMS-based normalization used in lieu of LayerNorm                               |
| Multi-Latent Attention | Enables efficient information routing between latent vectors                    |
| Mixture-of-Experts     | Switch-style expert selection; improves model capacity without bloating compute |

---

### 🧪 Example Output

**Input:**
> Input: "The capital of France is"

**Summary:**
> Output: "Paris."

---

### 🫵 Running the Model

#### Before Usage:

Install your requirements
> pip install -r requirements.txt

Change your config parameters (model architecture and training parameters) to your liking in the config.yaml file
```yaml
model:
   num_embd: 128              # number of dimensions to represent the token
   num_layers: 6              # number of transformer blocks
   num_dense_layers: 2        # First N layers which are dense
   num_attention_heads: 4     # Number of attention heads per block (head_dimension = num_embd//num_attention heads)
   intermediate_size: 512     # Overall FeedForward dim size (4 * n_embd)

   # MLA
   latent_dim: 4              # Dimension of the compressed latent space (head dim//2 or 4)
   proj_matrix_size: 64       # the proj matrices to up and down project
   q_lora_rank: 0             # LoRA rank for query projections.
   kv_lora_rank: 0            # LoRA rank for key-value projections.
   qk_nope_head_dim: 32       # Dimension for query-key projections without positional embeddings.
   qk_rope_head_dim: 32       # Dimension for query-key projections with rotary embeddings.
   v_head_dim: 32             # Dimension for value projections.

   # MoE
   num_experts: 4             # Total number of routed experts per MoE layer
   num_shared_experts: 1      # Number of experts that get activated no matter the token
   num_activated_experts: 2   # K: num of experts activated per token
   expert_inter_size: 512     # Intermediate size for each expert's FFN (intermediate_size/num_experts * 2)
   n_expert_groups: 1         # Number of groups experts are grouped in
   n_limited_groups: 1        # Number of experts per group
   route_scale: 1             # Scaling factor per route
   score_function: "sigmoid"  # The scoring function used to determine what expert/s to route to in MOE

   # RoPE
   original_seq_len: 128      # Original sequence length
   rope_theta: 10000.0        # Base for rotary positional encoding
   rope_factor: 10            # Scaling factor for extended sequence lengths
   beta_fast: 16              # Fast beta correction factor
   beta_slow: 1               # Slow beta correction factor
   mscale: 1                  # Scaling factor for extended attention

training:
   max_batch_size: 32         # Maximum number of batches trained at the same time
   num_epochs: 10             # number of complete passes through the dataset
   learning_rate: 0.0003      # learning rate for training
   warmup_steps: 500          # starting steps with low learning rate
   num_predicted_tokens: 1    # number of future tokens that are simultaneously predicted
   max_seq_len: 128           # Maximum sequence length.

data:
   vocab_size: 50257          # number of tokens in the vocabulary
   max_seq_len: 128           # Context size
   dtype: "bf16"              # Data type to represent floats

device:
   device: "cpu"              # Device to be used for training
   world_size: 1              # how many systems are you distributing across
```

##### Usage:

How to train the nano-DeepseekV3 model on the TinyStories Dataset?
> python model/train.py

How to run this model?
> python model/generate.py

### Architecture
![DeepseekV3 Architecture](/visualizations/Screenshot%202025-08-06%20at%203.17.07 PM.png)

DeepSeekV3-Mini is a compact transformer model inspired by the full-scale DeepSeek-V3 architecture. It incorporates several modern improvements to the transformer backbone to enable more efficient training, better contextual understanding, and smoother generalization. One such component is Rotary Positional Embeddings (RoPE), which replaces traditional positional encodings with a more elegant, rotation-based solution. RoPE injects position information directly into the attention mechanism and enables the model to generalize to longer sequences, thanks to its ability to extrapolate beyond the trained context window.

![MOE and MLA](/visualizations/Screenshot%202025-08-06%20at%203.14.33 PM.png)

Another key modification is the use of RMSNorm over the traditional LayerNorm. RMSNorm (Root Mean Square Normalization) removes the bias and mean-centering of inputs and instead normalizes inputs based on their root mean square value. This slight change results in improved training stability and efficiency while maintaining the same representational power. We also implement Multi-Latent Attention, a powerful technique introduced to allow the model to attend not just to token positions but to learn richer "latent slots" of information. These latents can dynamically route and combine signals, giving the model a better handle on abstract reasoning and token dependencies.

```py
class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.num_embd
        self.n_heads = config.num_attention_heads
        self.latent_dim = config.latent_dim  # e.g. 16
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, "Embedding dim must be divisible by number of heads"

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        # Latent compression: reduce head_dim to latent_dim
        self.compress_q = nn.Linear(self.head_dim, self.latent_dim)
        self.compress_k = nn.Linear(self.head_dim, self.latent_dim)
        self.compress_v = nn.Linear(self.head_dim, self.latent_dim)

        # Latent decompression: bring it back to head_dim
        self.decompress = nn.Linear(self.latent_dim, self.head_dim)

        self.softmax_scale = 1.0 / math.sqrt(self.latent_dim)

    def forward(self, x: torch.Tensor, start_pos=0, freqs_cis=None, mask=None):
        B, T, D = x.size()
        H = self.n_heads
        Hd = self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd)
        k = self.k_proj(x).view(B, T, H, Hd)
        v = self.v_proj(x).view(B, T, H, Hd)

        # Slice rotary embeddings to current sequence length T
        if freqs_cis is not None:
            freqs_cis = freqs_cis[start_pos : start_pos + T, :]

        q, k = apply_rope_embeddings(q, k, freqs_cis)
        q_latent = self.compress_q(q)
        k_latent = self.compress_k(k)
        v_latent = self.compress_v(v)

        scores = torch.einsum("bTHl,bSHl->bHTS", q_latent, k_latent) * self.softmax_scale

        if mask is not None:
            expanded_mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + expanded_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum("bhts,bshl->bthl", attn_weights, v_latent)

        attn_output = self.decompress(attn_output)
        attn_output = attn_output.contiguous().view(B, T, D)
        return self.out_proj(attn_output)
```

To enhance parameter efficiency and model scalability, we add two more important components. First, Mixture of Experts (MoE) allows different parts of the model to specialize on different types of inputs. Instead of using all feedforward layers for every token, a gating mechanism decides which experts to activate, making computation both sparse and targeted. Second, we integrate LoRA (Low-Rank Adaptation) into the attention layers, enabling efficient fine-tuning by only updating a small number of parameters. This makes it possible to adapt the model to new tasks or domains without retraining the full parameter set — a practical solution when working with limited resources.

```py
class Router(nn.Module):
    def __init__(self, config: DPS_Config):
        super().__init__()
        self.num_embd = config.num_embd
        self.num_activated_experts = config.num_activated_experts
        self.num_expert_groups = config.n_expert_groups
        self.num_limited_groups = config.n_limited_groups
        self.score_function = config.score_function
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.num_embd))
        self.bias = nn.Parameter(torch.empty(config.num_experts)) if self.num_embd == 1024 else None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = F.linear(x, self.weight, self.bias)
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_function == "softmax" else scores.sigmoid()
        original_scores = scores
        indices = torch.topk(scores, self.num_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        if self.score_function == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        weights = weights * self.route_scale
        return weights.type_as(x), indices

class MOE(nn.Module):
    def __init__(self, config: DPS_Config):
        super().__init__()
        self.config = config
        self.router = Router(config)
        self.experts = nn.ModuleList([MLP(config.num_embd, config.expert_inter_size) for i in range(config.num_experts)])
        self.shared_experts = MLP(config.num_embd, config.num_shared_experts * config.expert_inter_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.config.num_embd) 
        weights, indices = self.router(x)
        y = torch.zeros_like(x)

        for index in range(x.size(0)):
            for top in range(weights.size(1)): 
                expert_idx = indices[index, top].item()
                weight = weights[index, top]
                expert = self.experts[expert_idx]
                y[index] = y[index] + expert(x[index].unsqueeze(0)).squeeze(0) * weight

        z = self.shared_experts(x)
        return (y + z).view(shape)
```

#### Overview
https://arxiv.org/pdf/2505.09343

#### Training:
- I used the FineWebEDU-10BT dataset to train my nano-DeepseekV3 model on high quality data. It is distilled from the FineWeb dataset and used the LLama3-70B-Instruct model to classify content in the Fineweb dataset as education quality. Using this, while we will not have the resources for the model to create generalized generations, we can make sure that the model memorizes higher quality data.
!(FineWebEDU)[https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/QqXOM8h_ZjjhuCv71xmV7.png]

- I began by overfitting on one data sample to see if the model would learn at all:
![overfitting loss curve](/visualizations/overfit_loss_curve.png)

- Then I began overfitting on multiple samples to see if the model can learn multiple pieces of information at the same time
![overfitting loss curve](/visualizations/small_batch_loss_curve.png)

- Then I trained on the tinystories dataset to see if the model would reliably learn on a small(er) dataset
