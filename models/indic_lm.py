import torch
import torch.nn as nn


from models.embeddings import (
    TokenEmbedding,
    PositionalEmbedding
)

from models.transformer import (
    TransformerBlock
)



class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_layers,
        num_heads,
        max_length,
        dropout=0.1
    ):

        super().__init__()


        self.embedding_dim = embedding_dim


        # Token embeddings

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )


        # Position embeddings

        self.position_embedding = PositionalEmbedding(
            max_length,
            embedding_dim
        )


        # Transformer stack

        self.layers = nn.ModuleList(
            [

                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    max_length,
                    dropout
                )

                for _ in range(num_layers)

            ]
        )


        # Final normalization

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )


        # Convert hidden state → vocabulary logits

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )



    def forward(self, idx):

        """
        Forward pass.


        idx:

        token IDs

        Shape:

        (batch, sequence_length)


        Example:

        [
          [12,45,78],
          [90,10,22]
        ]

        """

        batch_size, seq_length = idx.shape



        # Token embeddings

        token_embeddings = self.token_embedding(
            idx
        )


        # Position embeddings

        position_embeddings = self.position_embedding(
            token_embeddings
        )


        # Combine

        x = (
            token_embeddings
            +
            position_embeddings
        )



        # Transformer layers

        for layer in self.layers:

            x = layer(x)



        # Normalize

        x = self.final_norm(x)



        # Vocabulary prediction

        logits = self.output_head(x)


        return logits
