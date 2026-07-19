## Indic Multimodal Neural Machine Translation Model

A research-oriented implementation of an Indic multimodal neural machine translation system combining morphology-aware text processing with visual feature grounding.

The project explores:

- Transformer-based multimodal neural machine translation
- Morphology-aware subword tokenization for Indic languages
- Visual feature extraction and image-text fusion
- Distributed Transformer fine-tuning workflows
- Parameter-efficient fine-tuning using LoRA vs full fine-tuning.

---
# Goals


---
# Project Overview

Many Indic languages are:

- low-resource
- morphologically rich
- highly agglutinative

Traditional tokenization methods often fragment words excessively.

Example:

```
ಕರ್ನಾಟಕದಲ್ಲಿ
```

A poor tokenizer may produce:

```
ಕರ
ನಾ
ಟಕ
ದ
ಲ್ಲಿ
```

A morphology-aware tokenizer learns:

```
ಕರ್ನಾಟಕ
ದಲ್ಲಿ
```

which preserves linguistic structure and improves downstream performance.


---

# System Architecture


```
                   Text Input
                       |
                       |
              NER Entity Masking
                       |
                       |
             SentencePiece Tokenizer
                       |
                       |
              Text Token Embeddings
                       |
                       |
                       |
                       |                 Image Input
                       |                     |
                       |                 Detectron2
                       |                     |
                       |            Visual Region Features
                       |                     |
                       |             Feature Projection
                       |                     |
                       +----------+----------+
                                  |
                                  |
                         Multimodal Fusion
                    (Early Fusion / Late Fusion)
                                  |
                                  |
                         mBART Encoder
                                  |
                                  |
                 Multimodal Contextual Representations
                                  |
                                  |
                         mBART Decoder
                                  |
                                  |
              Causal Self-Attention + Cross-Attention
                                  |
                                  |
                         Linear + Softmax
                                  |
                                  |
                    Kannada Token Prediction
                                  |
                                  |
                         Kannada Translation
```


The Transformer architecture contains:

- token embeddings
- positional embeddings
- multi-head self-attention
- feed-forward layers
- causal masking
- autoregressive generation


Visual features provide additional context for ambiguous sentences.


Example:


Text:

```
The player picked up the bat.
```


Without image:

```
bat = animal ?
```

With image:

```
cricket bat detected
```

The model can select the correct meaning.


---

# Features


## Model Components

- mBART encoder-decoder Transformer backbone
- SentencePiece tokenizer
- NER-based entity masking
- Detectron2 visual feature extraction
- Multimodal fusion layers
- LoRA/full fine-tuning support


## Tokenization Analysis

Evaluates:

- BPE
- WordPiece
- SentencePiece


Metrics:

- token fragmentation ratio
- average tokens per word
- morphology preservation


---

## Training Strategy

Implemented:

- mBART fine-tuning
- Detectron2 visual feature extraction
- Image-text fusion
- NER-based entity masking


---

## Parameter Efficient Fine-Tuning

Supports:

### Full Fine-tuning

Updates:

```
All model parameters
```


### LoRA

Updates:

```
Small trainable adapter matrices
```

while freezing the base model.

---

# Training Pipeline


```
Dataset
   |
Preprocessing
   |
Tokenizer
   |
DataLoader
   |
Distributed Training
   |
Checkpoint Saving
   |
Evaluation
```


Supports:

- PyTorch DistributedDataParallel
- mixed precision training
- checkpoint recovery
- experiment tracking

---

# Installation


Clone repository:


```bash
git clone https://github.com/<username>/indic-multimodal-llm.git

cd indic-multimodal-llm
```


Create environment:


```bash
conda create -n indicllm python=3.10

conda activate indicllm
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

# Training Multimodal Translation Model

Prepare dataset:
python data/preprocessing.py

Train tokenizer:
python tokenizer/train_tokenizer.py

Extract image features:
python data/image_features.py

Fine-tune mBART:
python train_mbart.py


Prepare dataset:

```bash
python data/preprocessing.py
```


Train tokenizer:

```bash
python tokenizer/train_tokenizer.py
```


Train model:

```bash
python train_llm.py
```


---

# Multimodal Translation Training


Extract image features:


```bash
python data/image_features.py
```


Fine-tune mBART:


```bash
python train_mbart.py
```


---

# Evaluation


Translation evaluation:


```bash
python evaluation/bleu.py
```


Metrics:

- BLEU
- METEOR
- ROUGE


---

# Results


Example benchmark:


| Model | BLEU |
|---|---:|
| Text-only mBART | 38.84 |
| Multimodal mBART | 40.51 |


The improvement demonstrates the benefit of visual grounding for translation.


---

# Technologies


| Component | Technology |
|-|-|
| Deep Learning | PyTorch |
| Language Models | HuggingFace Transformers |
| Tokenization | SentencePiece |
| Vision | Detectron2 |
| Fine-tuning | PEFT LoRA |
| Evaluation | SacreBLEU |
| Distributed Training | DDP |


---

# Future Improvements


- Larger Indic corpora
- Flash Attention integration
- FSDP training
- Retrieval-augmented generation
- Instruction tuning
- Preference optimization


---

# License

MIT License
