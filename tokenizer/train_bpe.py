"""
Train BPE tokenizer.

Used for tokenizer comparison experiments.
"""


from tokenizers import Tokenizer

from tokenizers.models import BPE

from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import Whitespace



INPUT_FILE = (
    "data/indic/processed/train.txt"
)


OUTPUT_FILE = (
    "tokenizer/bpe_tokenizer.json"
)



def train_bpe():


    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )


    tokenizer.pre_tokenizer = (
        Whitespace()
    )


    trainer = BpeTrainer(

        vocab_size=32000,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]"
        ]

    )


    tokenizer.train(

        [INPUT_FILE],

        trainer

    )


    tokenizer.save(
        OUTPUT_FILE
    )



if __name__ == "__main__":

    train_bpe()
