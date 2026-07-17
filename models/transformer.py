"""
Transformer Block implementation.

A GPT-style Transformer block contains:

1. Layer Normalization
2. Multi-Head Self-Attention
3. Residual Connection
4. Feed Forward Network
5. Residual Connection


Architecture:

        x
        |
        |
   LayerNorm
        |
        |
 MultiHead Attention
        |
        |
       (+)
        |
        |
   LayerNorm
        |
        |
 Feed Forward Network
        |
        |
       (+)
        |
        |
      output

"""

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward



class TransformerBlock(nn.Module):
    """
    Single GPT Transformer block.
    """


    def __init__(
        self,
        embedding_dim,
        num_heads,
        max_length,
        dropout=0.1
    ):

        super().__init__()


        # Normalize before attention

        self.layer_norm1 = nn.LayerNorm(
            embedding_dim
        )


        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            max_length
        )


        self.dropout1 = nn.Dropout(
            dropout
        )


        # Normalize before FFN

        self.layer_norm2 = nn.LayerNorm(
            embedding_dim
        )


        self.feed_forward = FeedForward(
            embedding_dim,
            dropout=dropout
        )


        self.dropout2 = nn.Dropout(
            dropout
        )



    def forward(self,x):

        """
        x:

        (batch, sequence, embedding_dim)

        """


        # =========================
        # Attention + Residual
        # =========================


        attention_output = self.attention(
            self.layer_norm1(x)
        )


        x = x + self.dropout1(
            attention_output
        )


        # =========================
        # FFN + Residual
        # =========================


        ff_output = self.feed_forward(
            self.layer_norm2(x)
        )


        x = x + self.dropout2(
            ff_output
        )


        return x
