# Data Pipeline


## Indic Language Model


                  DOWNLOAD

AI4Bharat IndicCorp              Flickr30k
        |                            |
        v                            v

data/indic/raw              data/multimodal/images
        |                            |
        |                            |
        v                            v

prepare_dataset.py          vision_features.py

        |                            |

        v                            v

data/indic/processed        data/multimodal/features

        |                            |

        v                            v

SentencePiece              multimodal mBART
Transformer                training
## Multimodal Translation


Images are processed offline using Detectron2.

Pipeline:

Images

↓

Detectron2 region features

↓

Multimodal dataset

↓

mBART fine-tuning


## Dataset classes

dataset.py

Creates:

input_ids → target_ids


multimodal_dataset.py

Creates:

visual features + source text + target text
