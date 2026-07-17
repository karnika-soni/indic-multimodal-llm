"""
Feed Forward Network module.

Every Transformer block contains:

1. Self Attention
2. Feed Forward Network


Attention mixes information across tokens.

FFN transforms each token representation independently.
"""

import torch
import torch.nn as nn



class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Formula:

    FFN(x) = W2(activation(W1x + b1)) + b2


    Input:

    (batch, seq_len, embedding_dim)


    Output:

    (batch, seq_len, embedding_dim)

    """

    def __init__(
        self,
        embedding_dim,
        expansion_factor=4,
        dropout=0.1
    ):

        super().__init__()


        hidden_dim = (
            embedding_dim 
            * expansion_factor
        )


        self.network = nn.Sequential(

            # Expand representation

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),


            # Non-linearity

            nn.GELU(),


            # Drop some neurons during training

            nn.Dropout(
                dropout
            ),


            # Project back

            nn.Linear(
                hidden_dim,
                embedding_dim
            ),

        )



    def forward(self,x):

        return self.network(x)
