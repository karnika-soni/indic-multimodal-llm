"""
Multimodal mBART Model.

Combines:

- Text input
- Visual region features
- Fusion layer
- mBART encoder-decoder


Task:

Image + Text -> Translation

"""

import torch
import torch.nn as nn

from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer
)


from multimodal.fusion import (
    VisualProjection,
    EarlyFusion
)




class MultimodalMBart(nn.Module):


    def __init__(
        self,
        model_name,
        visual_dim=1024,
        hidden_dim=1024
    ):

        super().__init__()



        self.mbart = (
            MBartForConditionalGeneration
            .from_pretrained(
                model_name
            )
        )



        self.visual_projection = (
            VisualProjection(
                visual_dim,
                hidden_dim
            )
        )



        self.fusion = EarlyFusion()



    def forward(
        self,
        input_ids,
        attention_mask,
        visual_features,
        labels=None
    ):

        """
        Forward pass.


        Text:

        input_ids

        Image:

        visual_features


        Output:

        translation logits

        """


        # Convert image features
        # into text embedding space

        visual_embeddings = (
            self.visual_projection(
                visual_features
            )
        )



        # Get text embeddings

        text_embeddings = (
            self.mbart
            .model
            .encoder
            .embed_tokens(
                input_ids
            )
        )



        # Combine modalities

        fused_embeddings = self.fusion(
            text_embeddings,
            visual_embeddings
        )



        outputs = self.mbart(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )


        return outputs
