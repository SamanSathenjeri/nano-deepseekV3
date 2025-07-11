import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from utils import load_config
import sys

@dataclass
class DPS_Config:
    """
    Defines model hyperparameters and arguments from config.yaml file

    Attributes:
        num_embd(int) = Number of dimensions to represent tokens
        num_layers(int) = Number of transformer blocks
        num_dense_layers(int) = Number of dense layers in the model (layers before we send to MoE)
        num_attention_heads(int) = Number of attention heads
        intermediate_size(int) = Intermediate dimension for MLP layers
        block_size(int) = Context size
        vocab_size(int) = Vocabulary size
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

    # MoE
    num_experts = config['model']['num_experts']
    num_activated_experts = config['model']['num_activated_experts']
    expert_inter_size = config['model']['expert_inter_size']

    # training
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    num_predicted_tokens = config['training']['num_predicted_tokens']

    # data
    block_size = config['data']['block_size']
    vocab_size = config['data']['vocab_size']
    train_file = config['data']['train_file']

    # device
    device = config['device']['device']
    world_size = config['device']['world_size']
    rank = config['device']['rank']

class Embedding(nn.Module):

    def __init__(self, vocab_size: int, num_embd: int, world_size: int, rank: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embd = num_embd
        self.world_size = world_size
        self.rank = rank

        assert self.vocab_size % self.world_size == 0, f"Vocabulary size must be divisible by world size (world_size={self.world_size})"
        self.part_vocab_size = (self.vocab_size // self.world_size)

        self.vocab_start_index = self.rank * self.part_vocab_size   
        self.vocab_end_index = self.vocab_start_index + self.part_vocab_size

        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.num_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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
    
class MLA(nn.Module):

class MOE(nn.Module):

class MLP(nn.Module):

class RMSNorm(nn.Module):

class OutputProjection(nn.Module):

class CreateRotaryEmbeddings(nn.Module):

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
            embed = Embedding(config.vocab_size, config.num_embd, config.world_size, config.rank), # creates the embedding
            layers = nn.ModuleList([Block(i, config) for i in range(config.num_layers)]), # creates num_layers of transformer blocks
            rms_norm = RMSNorm(config.num_embd), # adds in an RMS norm at the end, after all of the transformer blocks
        ))

        self.output_proj = OutputProjection(config.vocab_size, config.num_embd)
        self.register_buffer("rotary_embeddings", CreateRotaryEmbeddings(self.config), persistent=False)

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
        embedded_tokens = self.embed(tokens)
        rotary_embeddings = self.rotary_embeddings[start_pos:start_pos+sequence_length]
        mask = None

        # creates mask of upper triangle with -inf elements, if sequence length is above the trivial amount of 1
        # to avoid the model from using future tokens to help it calculate the attention (the triu_ is an upper triangle mask and is an inplace operation)
        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float("-inf"), device=tokens.device).triu_(1)
        
        # takes in the output of every layer
        for layer in self.layers:
            embedded_tokens = layer(embedded_tokens, start_pos, rotary_embeddings, mask)
        
        # normalizes the output and down projects to logits
        embedded_tokens = self.norm(embedded_tokens)[:, -1]
        logits = self.output_proj(embedded_tokens)

        return logits        
    
if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16) #sets the datatype to Bfloat 16 (shortened mantissa)
    torch.set_default_device("cuda") #sets the device to cuda
    torch.manual_seed(0)
    config = DPS_Config() # creates a new model Config
    x = torch.randint(0, config.vocab_size, (2, 128)) # creates a sample tensor of random values with size (2, 128)
    model = DPS(config) # creates a model using the config
    print(model(x).size()) 