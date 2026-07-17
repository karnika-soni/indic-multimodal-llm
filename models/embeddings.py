"""
Embedding layers for Indic LLM.

Converts:
    
Token IDs

    ↓

Dense vectors

Example:

ಕರ್ನಾಟಕ

token id:
1523

↓

embedding:

[
0.23,
-0.11,
0.56,
...
]

"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into learned embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )


    def forward(self, x):
        """
        Input:

        x:
        (batch_size, sequence_length)

        Example:

        (32, 512)


        Output:

        (batch_size, sequence_length, embedding_dim)

        Example:

        (32,512,512)

        """

        return self.embedding(x)



class PositionalEmbedding(nn.Module):
    """
    Adds information about token position.

    Transformer has no recurrence,
    so it does not know word order.

    Example:

    "ರಾಮನು ಮನೆಗೆ ಹೋದನು"

    and

    "ಮನೆಗೆ ರಾಮನು ಹೋದನು"

    have same tokens but different meaning.

    Position embeddings solve this.
    """

    def __init__(
        self,
        max_length: int,
        embedding_dim: int
    ):
        super().__init__()


        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )


    def forward(self, x):

        batch_size, seq_length = x.shape[:2]


        positions = torch.arange(
            seq_length,
            device=x.device
        )


        positions = positions.unsqueeze(0)


        return self.position_embedding(
            positions
        )
