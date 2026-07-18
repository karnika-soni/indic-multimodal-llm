"""
Data loading utilities.

Loads raw Indic text corpora
for decoder-only language model training.
"""


from pathlib import Path



def load_text_file(path):

    """
    Load text corpus.

    Expected format:

    one document/sentence per line

    """

    path = Path(path)


    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        texts = [
            line.strip()
            for line in f
            if line.strip()
        ]


    return texts
