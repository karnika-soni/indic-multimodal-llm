"""
Dataset download utilities.

Downloads:

1. Indic text corpus
   - used for decoder-only Indic LLM training

2. Flickr30k image-text dataset
   - used for multimodal mBART fine-tuning


The script only downloads raw data.
Preprocessing happens separately.

"""


from pathlib import Path

from datasets import load_dataset



# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path("data")


INDIC_DIR = (
    DATA_DIR /
    "indic" /
    "raw"
)


MULTIMODAL_DIR = (
    DATA_DIR /
    "multimodal"
)



# -----------------------------
# Create folders
# -----------------------------


def create_dirs():

    INDIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    MULTIMODAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



# -----------------------------
# Download Indic corpus
# -----------------------------


def download_indic_dataset():

    """
    Downloads Indic text corpus.

    Note:
    Replace dataset identifier
    depending on selected IndicCorp release.
    """


    print(
        "Downloading Indic corpus..."
    )


    dataset = load_dataset(
        "ai4bharat/IndicCorp"
    )


    dataset.save_to_disk(
        INDIC_DIR
    )


    print(
        "Indic corpus saved:",
        INDIC_DIR
    )



# -----------------------------
# Download Flickr30k
# -----------------------------


def download_flickr30k():

    """
    Downloads Flickr30k image-text pairs.

    Dataset contains:

    image
    captions

    """

    print(
        "Downloading Flickr30k..."
    )


    dataset = load_dataset(
        "nlphuji/flickr30k"
    )


    dataset.save_to_disk(
        MULTIMODAL_DIR
    )


    print(
        "Flickr30k saved:",
        MULTIMODAL_DIR
    )



# -----------------------------
# Main
# -----------------------------


def main():

    create_dirs()


    download_indic_dataset()


    download_flickr30k()



if __name__ == "__main__":

    main()
