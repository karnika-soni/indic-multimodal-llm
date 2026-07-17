"""
Multimodal fusion modules.

Combines:

Text embeddings

+

Visual embeddings


Supports:

- early fusion
- late fusion

"""

import torch
import torch.nn as nn




class VisualProjection(nn.Module):


    """
    Maps image features
    into text embedding space.
    """


    def __init__(
        self,
        visual_dim,
        text_dim
    ):

        super().__init__()


        self.projection = nn.Linear(
            visual_dim,
            text_dim
        )



    def forward(
        self,
        visual_features
    ):

        return self.projection(
            visual_features
        )





class EarlyFusion(nn.Module):

    """
    Concatenate visual tokens
    with text tokens.

    Example:


    [IMAGE TOKENS]

            +

    [TEXT TOKENS]


    fed into encoder together.

    """


    def forward(
        self,
        text_embeddings,
        visual_embeddings
    ):


        return torch.cat(
            [
                visual_embeddings,
                text_embeddings
            ],
            dim=1
        )





class LateFusion(nn.Module):

    """
    Combines representations
    after separate encoding.
    """


    def __init__(
        self,
        hidden_dim
    ):

        super().__init__()


        self.layer = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )



    def forward(
        self,
        text_features,
        visual_features
    ):


        combined = torch.cat(
            [
                text_features,
                visual_features
            ],
            dim=-1
        )


        return self.layer(
            combined
        )
