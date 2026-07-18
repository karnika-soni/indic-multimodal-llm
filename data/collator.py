"""
Batch collation utilities.

Pads variable length sequences
for Transformer training.
"""


import torch

from torch.nn.utils.rnn import pad_sequence




class LanguageModelCollator:


    def __init__(
        self,
        pad_token_id
    ):

        self.pad_token_id = (
            pad_token_id
        )



    def __call__(
        self,
        batch
    ):


        inputs = [

            item["input"]

            for item in batch

        ]


        targets = [

            item["target"]

            for item in batch

        ]



        inputs = pad_sequence(

            inputs,

            batch_first=True,

            padding_value=
            self.pad_token_id

        )


        targets = pad_sequence(

            targets,

            batch_first=True,

            padding_value=-100

        )


        return {


            "input":

            inputs,


            "target":

            targets

        }
