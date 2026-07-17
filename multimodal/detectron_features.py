"""
Detectron2 feature extraction.

Converts images into region-level
visual embeddings.

Image:

(H,W,3)

↓

Detectron2 backbone

↓

Region features:

(num_regions, feature_dim)

"""

import torch
import torch.nn as nn



class VisualFeatureExtractor(nn.Module):


    def __init__(
        self,
        backbone,
        feature_dim=1024
    ):

        super().__init__()


        self.backbone = backbone

        self.feature_dim = feature_dim



    def forward(
        self,
        images
    ):

        """
        images:

        batch of images


        Output:

        visual embeddings


        Shape:

        batch,
        regions,
        feature_dim

        """


        features = self.backbone(
            images
        )


        return features
