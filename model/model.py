import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from utils import load_config
from MLA import MLA
from RoPE import precompute_rope_embeddings
# import sys

rank = 0

@dataclass
class DPS_Config:
    """
    Defines model hyperparameters and arguments from config.yaml file

    Attributes:
        
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
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    num_predicted_tokens = config['training']['num_predicted_tokens']
    max_seq_len = config['training']['max_seq_len']

    # data
    block_size = config['data']['block_size']
    vocab_size = config['data']['vocab_size']
    dtype = config['data']['dtype']

    # device
    device = config['device']['device']
    world_size = config['device']['world_size']

class Embedding(nn.Module):
    """
    The embedding layer that can be computed using parallel processes

    Attributes:
        vocab_size (int): Vocabulary size.
        num_embd (int): Number of dimensions to represent tokens
        world_size (int): Number of processes to parallelize on
        part_vocab_size (int): Initializes the number of tokens each process will keep in its respective vocabulary
        vocab_start_index (int): the starting index of the current processes vocabulary allocation
        vocab_end_index (int): the ending index of the current processes vocabulary allocation
        weight (nn.Parameter): The embedding matrix
    """
    def __init__(self, vocab_size: int, num_embd: int, world_size: int):
        """
        Initializing the embedding layer

        Args:
            vocab_size (int): Vocabulary size.
            num_embd (int): Number of dimensions to represent tokens
            world_size (int): Number of processes
        """
        super().__init__()

        # initializing local variables
        self.vocab_size = vocab_size
        self.num_embd = num_embd
        self.world_size = world_size

        # asserting that vocab_size can be equally divided by number of processes (to avoid messy calculations)
        assert self.vocab_size % self.world_size == 0, f"Vocabulary size must be divisible by world size (world_size={self.world_size})"
        self.part_vocab_size = (self.vocab_size // self.world_size)

        # assigns each process an equal part of the vocabulary (defines start and end index of vocabulary for this particular process)
        self.vocab_start_index = rank * self.part_vocab_size   
        self.vocab_end_index = self.vocab_start_index + self.part_vocab_size

        # creates a weight parameter that holds the assigned part of the dictionary
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.num_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass of the embedding layer

        Args:
            x (torch.Tensor): input tokens

        Returns: 
            y (torch.Tensor): (vocab_size, num_embd) size tensor of token embeddings
        """
        # splitting the embedding calculationg of parts of the vocab for this particular processor (if multiple processors)
        if self.world_size > 1:
            mask = (x < self.vocab_start_index) | (x > self.vocab_end_index)
            x = x - self.vocab_start_index
            x[mask] = 0

        # the translation of tokens to embeddings for this processor
        y = F.embedding(x, self.weight)

        # concatenating all the tokens from the various processors (if multiple processors)
        if self.world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)

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
        num_embd(int): Number of dimensions to represent tokens
        intermediate_size(int): Intermediate dimension 
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
        '''
        self.num_embd = num_embd
        self.intermediate_size = intermediate_size

        '''
        !!!more about the SWiGLU gating mechanism!!!
        the output from silu(layer1(x)) is a nonlinear activation 
        the output from layer3(x) is the raw output from the linear layer's transformations
        they are element-wise multiplied where the layer3(x)'s output acts as a "gate" for the output from silu(layer1(x))
        what happens is if the activations from layer3(x) is high, then it will amplify the corresponding feature from the silu
        else, if the activation from the layer3(x) is low, then it will diminish and "close the gate" of the corresponding 
        feature from the silu - this allows us highlight or hide certain features based on what our input is
        '''
        
        self.layer1 = nn.Linear(self.num_embd, self.intermediate_size) # Sets up linear layer to use SiLU
        self.layer2 = nn.Linear(self.intermediate_size, self.num_embd) # Sets up linear layer to project back to num_embd dimensions
        self.layer3 = nn.Linear(self.num_embd, self.intermediate_size) # Sets up linear layer to be used in SWiGLU gating mechanism

    def forward(self, x: torch.Tensor):
        '''
        Forward Pass of the MLP layer
        
        Arguments:
            x(torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: output tensor after MLP operations
        '''
        return self.layer2(F.silu(self.layer1(x)) * self.layer3(x))
    
class Router(nn.Module):

    def __init__(self):

    def forward():
        

class Expert(nn.Module):

    def __init__():

    def forward():
        

class MOE(nn.Module): 
    '''
    The mixture of experts module which will allow for the routing to different experts
    '''
    def __init__(self, config: DPS_Config):
        self.num_experts = config.num_experts
        self.num_activated_experts = config.num_activated_experts
        self.expert_inter_size = config.expert_inter_size
        self.num_shared_experts = config.num_shared_experts
        self.n_expert_groups = config.n_expert_groups
        self.n_limited_groups = config.n_limited_groups
        self.route_scale = config.route_scale

    def forward():

        
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
            config (Config): Input tensor of token IDs with shape (batch_size, seq_len)
        """

        super().__init__()
        self.config = config

        # creating a module dict to use keys to refer to dict values
        # dict consists of the initial up projection, the transformer blocks, the normalization, and then the down projection
        self.transformer = nn.ModuleDict(dict(
            embed = Embedding(config.vocab_size, config.num_embd, config.world_size), # creates the embedding
            layers = nn.ModuleList([Block(i, config) for i in range(config.num_layers)]), # creates num_layers of transformer blocks
            rms_norm = RMSNorm(config.num_embd), # adds in an RMS norm at the end, after all of the transformer blocks
            output_proj = nn.Linear(config.num_embd, config.vocab_size) # finally down projects to get final predictions
        ))

        self.register_buffer("rotary_embeddings", precompute_rope_embeddings(self.config), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):

        """
        Forward pass of the Transformer

        Arguments:
            tokens (torch.Tensor): input tensor of tokens with the shape of (batch_size, sequence_length)

        Returns:
            logits(torch.Tensor): Logits tensor of shape (batch_size, vocab_size)
        """

        # creates embedded tokens and rotary embeddings
        sequence_length = tokens.size(1)
        embedded_tokens = self.transformer.embed(tokens)
        rotary_embeddings = self.rotary_embeddings[start_pos:start_pos+sequence_length]
        mask = None

        # creates mask of upper triangle with -inf elements, if sequence length is above the trivial amount of 1
        # to avoid the model from using future tokens to help it calculate the attention (the triu_ is an upper triangle mask and is an inplace operation)
        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float("-inf"), device=tokens.device).triu_(1)
        
        # takes in the output of every layer
        for layer in self.transformer.layers:
            embedded_tokens = layer(embedded_tokens, start_pos, rotary_embeddings, mask)
        
        # normalizes the output and down projects to logits
        embedded_tokens = self.transformer.rms_norm(embedded_tokens)[:, -1]
        logits = self.transformer.output_proj(embedded_tokens)

        return logits        
    
if __name__ == "__main__":
    config = DPS_Config() # creates a new model Config
    torch.set_default_dtype(torch.bfloat16) #sets the datatype to Bfloat 16 (shortened mantissa)
    torch.set_default_device(config.device)
    torch.manual_seed(0)
    x = torch.randint(0, config.vocab_size, (2, 128)) # creates a sample tensor of random values with size (2, 128)
    model = DPS(config) # creates a model using the config
    print(model(x).size()) 