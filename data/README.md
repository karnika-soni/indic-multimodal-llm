# Data Pipeline


## Indic Language Model


Pipeline:

Raw Indic corpus

↓

Cleaning and preprocessing

↓

SentencePiece/BPE/WordPiece evaluation

↓

Selected tokenizer

↓

Autoregressive dataset


Structure:

data/

├── indic/

│   ├── raw/

│   └── processed/


├── multimodal/

│   ├── images/

│   ├── metadata.json

│   └── features/


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
