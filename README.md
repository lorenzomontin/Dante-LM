# Dante- LM

A hands-on exploration of language modeling through Dante Alighieri's *Divine Comedy*. The project uses the poem, considered one of the greatest works of Western literature and the foundation of the modern Italian language, as a fixed testbed for building, training, and comparing increasingly capable neural network architectures.

Just as the poem follows a journey from Hell to Paradise, this repo tracks a personal journey through deep learning: starting from a simple MLP with no library abstractions, moving to a from-scratch Transformer, and finishing with fine-tuning a pretrained GPT. Each stage is meant to build intuition  and understanding before reaching for higher-level tools.



![alt text](image.png) 


## Model explored and tested:

| Model | Status | Notes |
|---|---|---|
| MLP (character/word-level, with BatchNorm) | Done | Implemented without `nn.Module` classes, to see the internal mechanics directly |
| Transformer (from scratch, character-level) | Done | Custom attention, multi-head, residual + LayerNorm blocks |
| GPT fine-tuning ([GePpeTto](https://huggingface.co/LorenzoDeMattei/GePpeTto)) | **In progress** | Fine-tuning an Italian pretrained GPT-2 on the *Commedia* |
| RNN | Planned | — |
| LSTM | Planned | — |


## Repository Structure 
- **EDA.ipnyb**: exploratory data analysis with tokenization and data splitting.
    Preprocessing decisions:
    - lowercase text (at least for now) for simplicity.
    - words separated by apostrophe(') are kept together, e.g. ch'io.
    - punctuation symbols are treated separately, to gather further information on Dante's writing style.   
    - editorial parts dropped: headlines with roman numerals, publisher's notes, and blank separation lines. All of these can 'pollute' the training data. 
    - Cantos structure is kept: canto boundaries are recorded as token-index markers (not baked into the token stream itself). This is needed for two things: 
        1) splitting train/val/test by whole cantos rather than shuffling individual context-target windows, since overlapping windows from adjacent canto text would otherwise leak near-duplicate examples across the split; 
        2) preventing any single sliding window from straddling a canto seam, even after splitting by canto.

- **MLP.ipynb**: a first  baseline: MLP with BatchNorm. The network is build without `nn.Module`, so every matrix multiplication, embedding lookup, and BatchNorm update is explicit. This was done for building intuition about activation scales, initialization, and why naive weight init blows up logits. The generated text, while low-loss, is not linguistically coherent.

- **Trf.ipynb**: a character-level Transformer (multi-head self-attention, residual connections, LayerNorm, feed-forward blocks, all implemented manually) trained directly on raw characters rather than tokenized words. Outperforms the MLP baseline and produces more Dante-*like* text, though grammar and rhyme are still inconsistent. See `experiments.md` for the full sweep across context length, embedding size, and depth.

- **GPT.ipynb** (ongoing): The final stage: moving from from-scratch models to fine-tuning GePpeTto, a GPT-2 model pretrained on Italian, to see how much a pretrained language prior improves fluency and rhyme compared to training from zero. Work in progress.

- **experiments.md**: hyperparameter tests and outcomes for Transformer and GPT.


## References 
- MLP-based language modeling: [Bengio et al.](https://www.alphaxiv.org/38e694ba-c6f9-490d-a699-cf0aae33db13)
- Italian GPT fine-tuning: [De Mattei et al.](https://www.alphaxiv.org/pdf/2004.14253v1)

## Setup
 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

