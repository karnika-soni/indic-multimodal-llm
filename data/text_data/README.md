# Indic Text Corpus


This directory stores processed Indic language text files.


Expected files:

- train.txt
- val.txt


Format:

One text sample per line.


Pipeline:

Raw corpus
→ cleaning
→ train/validation split
→ SentencePiece tokenization
→ autoregressive training samples
