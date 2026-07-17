"""
Multi-Head Self Attention module.

This is the core mechanism used in GPT.

Input:

(batch, sequence_length, embedding_dim)


Output:

(batch, sequence_length, embedding_dim)


Example:

32 sentences
32 tokens each
512 dimensional embeddings

Input:

(32,32,512)

Output:

(32,32,512)

"""

import torch
import torch.nn as nn



class SelfAttentionHead(nn.Module):
    """
    Single attention head.

    Computes:

    Attention(Q,K,V)

    = softmax(QK^T / sqrt(d))V

    """


    def __init__(
        self,
        embedding_dim,
        head_dim,
        max_length
    ):

        super().__init__()


        self.query = nn.Linear(
            embedding_dim,
            head_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            head_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            head_dim
        )


        # causal mask
        # prevents looking into future tokens

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    max_length,
                    max_length
                )
            )
        )


    def forward(self,x):

        """
        x:

        batch, seq_len, embedding_dim

        """


        batch_size, seq_len, _ = x.shape


        # Generate Query, Key, Value

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)



        # Attention scores

        scores = Q @ K.transpose(
            -2,
            -1
        )


        # scale

        scores = scores / (
            K.shape[-1] ** 0.5
        )



        # causal masking

        mask = self.mask[
            :seq_len,
            :seq_len
        ]


        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )



        # convert scores to probabilities

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        # weighted aggregation

        output = attention_weights @ V


        return output




class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads.

    Instead of learning one relationship,
    GPT learns multiple relationships.

    Example:

    Head 1:
    subject relationships

    Head 2:
    grammar

    Head 3:
    long distance dependency

    """


    def __init__(
        self,
        embedding_dim,
        num_heads,
        max_length
    ):

        super().__init__()


        assert embedding_dim % num_heads == 0


        head_dim = embedding_dim // num_heads



        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_dim,
                    head_dim,
                    max_length
                )

                for _ in range(num_heads)
            ]
        )


        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )



    def forward(self,x):


        # Run every attention head

        outputs = [
            head(x)
            for head in self.heads
        ]


        # concatenate heads

        x = torch.cat(
            outputs,
            dim=-1
        )


        # combine information

        x = self.projection(x)


        return x
