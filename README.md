# Indic Multimodal Language Model

A research-oriented implementation of an Indic Large Language Model (LLM) and Multimodal Neural Machine Translation system.

The project explores:

- Transformer-based Indic language modeling
- Morphology-aware subword tokenization
- Distributed large-scale pretraining workflows
- Multimodal machine translation using image features
- Parameter-efficient fine-tuning using LoRA vs full fine-tuning.


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


## 1. Indic Language Model

```
Indic Corpus
      |
      |
SentencePiece Tokenizer
      |
      |
Token IDs
      |
      |
Transformer Decoder
      |
      |
Next Token Prediction
```


The language model contains:

- token embeddings
- positional embeddings
- multi-head self-attention
- feed-forward layers
- causal masking
- autoregressive generation


---

# 2. Multimodal Translation Architecture


```
                 Image
                   |
                   |
              Detectron2
                   |
                   |
          Visual Region Features
                   |
                   |
           Feature Projection
                   |
                   |
Text ---------> mBART Encoder
                   |
                   |
              Decoder
                   |
                   |
            Translation Output

```


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


## Indic LLM Training

- Transformer decoder architecture
- Kannada / Indic corpus support
- SentencePiece tokenization
- Autoregressive training
- GPU accelerated training


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

## Multimodal Neural Machine Translation

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


### QLoRA

Uses:

```
4-bit quantized weights
+
LoRA adapters
```


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

# Repository Structure


```
indic-multimodal-llm/

├── configs/
│
├── models/
│   ├── attention.py
│   ├── transformer.py
│   ├── tiny_gpt.py
│   └── mbart_multimodal.py
│
├── tokenizer/
│
├── data/
│
├── training/
│
├── evaluation/
│
├── train_llm.py
├── train_mbart.py
└── generate.py

```


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

# Training Indic Language Model


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
| Fine-tuning | PEFT LoRA / QLoRA |
| Evaluation | SacreBLEU |
| Distributed Training | DDP |


---

# Future Improvements


- Larger Indic corpora
- Flash Attention integration
- FSDP training
- Retrieval augmented generation
- Instruction tuning
- Preference optimization


---

# License

MIT License
