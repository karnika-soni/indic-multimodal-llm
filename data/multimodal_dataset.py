"""
Multimodal dataset for mBART training.

Each sample contains:

- image region features extracted using Detectron2
- source language text
- target translation text


Example:

Image:
    street_scene.jpg


Detectron2 features:

    [num_regions, feature_dim]


Source:

    "A man riding a bicycle"


Target:

    "ಒಬ್ಬ ವ್ಯಕ್ತಿ ಸೈಕಲ್ ಓಡಿಸುತ್ತಿದ್ದಾನೆ"


Returns:

{
    image_features,
    input_ids,
    attention_mask,
    labels
}

"""


import json

from pathlib import Path

import torch

from torch.utils.data import Dataset




class MultimodalDataset(Dataset):


    def __init__(
        self,
        metadata_file,
        feature_dir,
        tokenizer,
        max_length=128
    ):


        self.tokenizer = tokenizer

        self.max_length = max_length


        self.feature_dir = Path(
            feature_dir
        )


        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.samples = json.load(f)



    def __len__(self):

        return len(
            self.samples
        )



    def __getitem__(
        self,
        idx
    ):


        sample = self.samples[idx]



        # -------------------------
        # Load visual features
        # -------------------------


        feature_path = (
            self.feature_dir
            /
            sample["feature_file"]
        )


        image_features = torch.load(
            feature_path
        )



        # Example:

        # image_features.shape

        # [
        #    num_regions,
        #    2048
        # ]



        # -------------------------
        # Tokenize source text
        # -------------------------


        source = (
            sample["source_text"]
        )


        target = (
            sample["target_text"]
        )



        source_tokens = (
            self.tokenizer(
                source,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        )



        target_tokens = (
            self.tokenizer(
                target,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        )



        input_ids = (
            source_tokens
            ["input_ids"]
            .squeeze(0)
        )


        attention_mask = (
            source_tokens
            ["attention_mask"]
            .squeeze(0)
        )


        labels = (
            target_tokens
            ["input_ids"]
            .squeeze(0)
        )



        # Ignore padding in loss

        labels[
            labels == self.tokenizer.pad_token_id
        ] = -100



        return {


            "image_features":
                image_features,


            "input_ids":
                input_ids,


            "attention_mask":
                attention_mask,


            "labels":
                labels

        }
