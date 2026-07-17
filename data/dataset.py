"""
Dataset loader for autoregressive language modeling.

Converts:

text

into:

input tokens

and

next-token targets.


Example:

Text:

ನಾನು ಮನೆಗೆ ಹೋದೆ


Input:

ನಾನು ಮನೆಗೆ


Target:

ಮನೆಗೆ ಹೋದೆ

"""

import torch

from torch.utils.data import Dataset



class IndicTextDataset(Dataset):


    def __init__(
        self,
        texts,
        tokenizer,
        max_length
    ):

        self.samples = []


        self.tokenizer = tokenizer

        self.max_length = max_length



        for text in texts:


            tokens = tokenizer.encode(
                text
            )


            # truncate

            tokens = tokens[
                :max_length + 1
            ]


            if len(tokens) > max_length:


                self.samples.append(
                    tokens
                )



    def __len__(
        self
    ):

        return len(
            self.samples
        )



    def __getitem__(
        self,
        idx
    ):


        tokens = self.samples[idx]


        input_ids = torch.tensor(
            tokens[:-1],
            dtype=torch.long
        )


        target_ids = torch.tensor(
            tokens[1:],
            dtype=torch.long
        )


        return {

            "input": input_ids,

            "target": target_ids

        }
