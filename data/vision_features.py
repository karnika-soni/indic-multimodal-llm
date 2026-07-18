"""
Prepare IndicCorp data.

Converts:

raw Indic corpus

into:

train.txt
val.txt
test.txt


used for tokenizer training.
"""


from pathlib import Path

import random



RAW_DIR = Path(
    "data/indic/raw"
)


OUTPUT_DIR = Path(
    "data/indic/processed"
)



def clean_text(text):

    return (
        text
        .strip()
    )



def prepare_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    texts = []


    for file in RAW_DIR.rglob("*"):


        if file.suffix not in [
            ".txt",
            ".json"
        ]:

            continue


        with open(
            file,
            encoding="utf-8"
        ) as f:


            for line in f:


                line = clean_text(
                    line
                )


                if line:

                    texts.append(
                        line
                    )



    random.shuffle(
        texts
    )


    n = len(texts)


    train_end = int(
        0.8*n
    )

    val_end = int(
        0.9*n
    )


    splits = {

        "train.txt":
            texts[:train_end],

        "val.txt":
            texts[train_end:val_end],

        "test.txt":
            texts[val_end:]

    }



    for name,data in splits.items():


        with open(
            OUTPUT_DIR/name,
            "w",
            encoding="utf-8"
        ) as f:


            f.write(
                "\n".join(data)
            )



    print(
        "Saved processed dataset:",
        OUTPUT_DIR
    )



if __name__ == "__main__":

    prepare_dataset()
