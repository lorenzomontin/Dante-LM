# Dante- LM

A hands-on exploration of language modeling through Dante Alighieri's *Divine Comedy*. The project uses the poem, considered one of the greatest works of Western literature and the foundation of the modern Italian language, as a fixed testbed for building, training, and comparing increasingly capable neural network architectures. The project is a work in progress, and the codebase may change over time.

Just as the poem follows a journey from Hell to Paradise, this repo tracks a personal journey through deep learning: starting from a simple MLP with no library abstractions, moving to a from-scratch Transformer, and finishing with fine-tuning a pretrained GPT. Each stage is meant to build intuition  and understanding before reaching for higher-level tools.



<img width="1600" height="1096" alt="Image" src="https://github.com/user-attachments/assets/a44eca2a-f21b-4acc-a8df-2047ca0e7b42" />


## Results 
 
The best model so far is [GePpeTto](https://huggingface.co/LorenzoDeMattei/GePpeTto) (an Italian GPT-2, ~109M parameters) fine-tuned with LoRA. It picks up Dante's vocabulary, archaic spellings and line layout while training 0.27% of the weights. It does not write terza rima yet.
 
| Model | Trainable params | Val loss | Val perplexity |
|---|---:|---:|---:|
| GePpeTto, zero-shot | 0 | 6.32 | ~554 |
| GePpeTto, full fine-tune (100 steps, batch 4, lr 5e-5) | ~109M | 4.20 | ~67 |
| GePpeTto + LoRA (r=8, α=16, `c_attn`; 1000 steps, batch 8, lr 2e-4) | 294,912 (0.27%) | 4.14 | ~63 |
 
Losses are per BPE token, at the last logged step. The full fine-tune also has a test loss of 4.20 (perplexity 66.4). 

The best from-scratch character-level transformer (2 blocks, d_model 384, context 128) reaches a validation loss of 1.49 per character.However, per-character and per-token losses are not directly comparable.
 
A LoRA sample (prompt in the first line, temperature 0.8, top-k 50):
 
```
Amor che ne la mente mi ragiona
come li occhi tuoi e ne lo cielo,
  e tu 'l mio volto, che ti si volge,
m'ha, e ti rimembra, e tu, nel cuore,
non mi sentirai
che 'l men, e di nuovo 'l padre mi chiami".
  "Oimè, a me: qual fu l'ultimo
dammi, come tu mi vorresti
```


## Model explored and tested:

| Model | Status | Notes |
|---|---|---|
| MLP (character/word-level, with BatchNorm) | Done | Implemented without `nn.Module` classes, to see the internal mechanics directly |
| Transformer (from scratch, character-level) | Done | Custom attention, multi-head, residual + LayerNorm blocks |
| GPT fine-tuning ([GePpeTto](https://huggingface.co/LorenzoDeMattei/GePpeTto)) | **In progress** | Fine-tuning an Italian pretrained GPT-2 on the *Commedia* |
| RNN | Planned | — |
| LSTM | Planned | — |


## Repository
 
- **`EDA.ipynb`**: exploratory data analysis, tokenization and data splitting. Preprocessing decisions:
    - lowercase text (at least for now) for simplicity.
    - words separated by an apostrophe are kept together, e.g. *ch'io*.
    - punctuation symbols are treated separately, to keep information on Dante's writing style.
    - editorial parts dropped: headlines with roman numerals, publisher's notes and blank separator lines, since they can pollute the training data.
    - canto structure is kept: canto boundaries are recorded as token-index markers, not baked into the token stream. This allows splitting train/val/test by whole cantos instead of shuffling context-target windows (overlapping windows from adjacent text would leak near-duplicate examples across splits), and prevents a window from straddling a canto seam.
- **`mlp.ipynb`**: a first baseline, an MLP with BatchNorm built without `nn.Module`, so every matrix multiplication, embedding lookup and BatchNorm update is explicit. This was to build intuition about activation scales, initialization, and why naive weight init blows up logits. The generated text has a low loss but is not linguistically coherent.
- **`trf.ipynb`**: a character-level Transformer (multi-head self-attention, residual connections, LayerNorm, feed-forward blocks, all written by hand), trained on the cased text with a 90/5/5 split by position. It produces more Dante-like text than the MLP, though grammar and rhyme are still inconsistent.
- **`trf_v2.ipynb`**: the same architecture using `nn.MultiheadAttention`, used for the later runs (batch 64, more blocks, longer context, lower learning rate).
- **`GPT.ipynb`**: GePpeTto on the *Commedia*: zero-shot baseline, full fine-tuning and LoRA, with qualitative sampling for each.
- **`geppetto-dante-lora/`**: the trained LoRA adapter (~1 MB).
- **`experiments.md`**: hyperparameters and outcomes for the transformer and GePpeTto runs.
The text comes from [dmf.unicatt.it/~della/pythoncourse18/commedia.txt](https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt). In the transformer and GPT notebooks the canto headers and title lines are removed (551,846 → 534,889 characters).



## Main findings

- Zero-shot GePpeTto writes fluent modern Italian and nothing like Dante.
- Full fine-tuning and LoRA both bring validation perplexity down to the 60s and produce Dante-like vocabulary (*'l*, *ne la*, *sanza*), but the outputs are repetitive and the lines do not rhyme. Full fine-tuning ran 100 steps and LoRA 1000, so the two numbers do not yet compare the methods fairly.
- With LoRA the adapter is about 1 MB, and validation loss was still falling at the end of the run.
- In the from-scratch transformer, a wider model (d_model 128 → 384) helped the most, a longer context did not help, and five blocks overfit. The full log is in [`experiments.md`](experiments.md).


## Limitations
 
- Each number comes from a single run without a fixed seed, and the losses are estimated on a few random windows.
- The text is split by position (90/5/5), so validation and test come from the last part of the Paradiso.
- The sample prompts are lines from the training text.
- The LoRA test loss has not been computed yet.


## Setup
 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To generate with the LoRA adapter:
 
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
name = "LorenzoDeMattei/GePpeTto"
tokenizer = AutoTokenizer.from_pretrained(name)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(name), "geppetto-dante-lora")
model.eval()
 
ids = tokenizer("Amor che ne la mente mi ragiona", return_tensors="pt")["input_ids"]
with torch.no_grad():
    out = model.generate(input_ids=ids, max_new_tokens=150, do_sample=True,
                         temperature=0.8, top_k=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```


## Next steps
 
- A rhyme metric (share of tercets whose first and third lines rhyme) for the real text and each model, so rhyme is measured and not judged by eye.
- Evaluate every model on the full validation and test sets, and compare full fine-tuning and LoRA with the same training budget.
- RNN and LSTM baselines.



## References 
- MLP-based language modeling: [Bengio et al.](https://www.alphaxiv.org/38e694ba-c6f9-490d-a699-cf0aae33db13)
- Italian GPT fine-tuning: [De Mattei et al.](https://www.alphaxiv.org/pdf/2004.14253v1)
- LoRA: [Hu et al.](https://www.alphaxiv.org/pdf/2106.09685)
- Structure of the from-scratch models follows Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.
