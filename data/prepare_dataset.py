"""
Dataset preparation.

Responsibilities:

- clean raw text
- remove empty samples
- create train/validation split
"""


import random

from pathlib import Path



def clean_text(text):

    text = text.strip()

    return text



def prepare_dataset(
    input_file,
    output_dir,
    val_ratio=0.1
):


    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        texts = [
            clean_text(x)
            for x in f.readlines()
            if x.strip()
        ]


    random.shuffle(
        texts
    )


    split = int(
        len(texts)
        *
        (1-val_ratio)
    )


    train = texts[:split]

    val = texts[split:]



    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        exist_ok=True
    )



    with open(
        output_dir/"train.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(train)
        )



    with open(
        output_dir/"val.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(val)
        )



if __name__ == "__main__":

    prepare_dataset(

        "raw.txt",

        "text_data"

    )
