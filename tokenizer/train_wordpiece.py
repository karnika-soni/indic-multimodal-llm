"""
Train WordPiece tokenizer.
"""


from tokenizers import Tokenizer

from tokenizers.models import WordPiece

from tokenizers.trainers import WordPieceTrainer

from tokenizers.pre_tokenizers import Whitespace



INPUT_FILE = (
    "data/indic/processed/train.txt"
)


OUTPUT_FILE = (
    "tokenizer/wordpiece_tokenizer.json"
)



def train_wordpiece():


    tokenizer = Tokenizer(

        WordPiece(
            unk_token="[UNK]"
        )

    )


    tokenizer.pre_tokenizer = (
        Whitespace()
    )


    trainer = WordPieceTrainer(

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

    train_wordpiece()
