# Dante-LM

A hands-on exploration of language modeling through Dante Alighieri's *Divine Comedy*. The project uses the poem, considered one of the greatest works of Western literature and the foundation of the modern Italian language, as a fixed testbed for building, training, and comparing increasingly capable neural network architectures. The project is a work in progress, and the codebase may change over time.

Just as the poem follows a journey from Hell to Paradise, this repo tracks a personal journey through deep learning: starting from a simple MLP with no library abstractions, moving to a from-scratch Transformer, and finishing with fine-tuning a pretrained GPT. Each stage is meant to build intuition  and understanding before reaching for higher-level tools.



<img width="1600" height="1096" alt="Image" src="https://github.com/user-attachments/assets/a44eca2a-f21b-4acc-a8df-2047ca0e7b42" />


## Results 
 
The main achievement so far is [GePpeTto](https://huggingface.co/LorenzoDeMattei/GePpeTto) (an Italian GPT-2, ~109M parameters) fine-tuned with LoRA. It reaches the lowest validation loss of the runs below while training 0.27% of the weights, and it picks up Dante's vocabulary, archaic spellings and line layout. It does not write terza rima yet, and it has not been compared with full fine-tuning at the same training budget.
 
| Model | Trainable params | Val loss | Test loss (perplexity) |
|---|---:|---:|---:|
| GePpeTto, zero-shot | 0 | 6.30 | 6.31 (553) |
| GePpeTto, full fine-tune (100 steps, batch 4, lr 5e-5) | ~109M | 4.17 | 4.19 (66.0) |
| GePpeTto + LoRA (r=8, α=16, `c_attn`; 1000 steps, batch 8, lr 2e-4) | 294,912 (0.27%) | 4.15 | 4.13 (61.9) |

Losses are per BPE token. Val is the last logged value (step 90 and step 900). Val and test are scored over the full split in non-overlapping 128-token windows.

The best from-scratch character-level transformer (2 blocks, d_model 384, context 128) reaches a validation loss of 1.49 per character. The word-level MLP baseline reaches 6.31 per word (vocabulary of 12,002 words, different split). Per-character, per-word and per-token losses are not directly comparable.
 
A LoRA sample (prompt in the first line, temperature 0.8, top-k 50):
```
Amor che ne la mente mi ragiona
che la testa, come d'un piè del tutto la memoria,
e quella, che non mi pare, che li occhi più savi.
  E io che son stato io a fare la terra,
e che 'nna veggo che 'l mondo è stato,
con la morte a colui ch'è morto;
  perché è suo, se non lo è stato, il nome che si vede
più che altro a coloro che mi son venuti.
```


## Main findings
 
- Zero-shot GePpeTto writes fluent modern Italian and nothing like Dante.
- Full fine-tuning and LoRA both bring test perplexity down from ~553 to 66 and 62, and produce Dante-like vocabulary (*'l*, *ne la*, *sanza*), but the outputs are repetitive and the lines do not rhyme. Full fine-tuning ran 100 steps and LoRA 1000, so the two numbers do not yet compare the methods fairly.
- With LoRA the adapter is about 1 MB, and validation loss was nearly flat over the last 200 of the 1000 steps.
- In the from-scratch transformer, a wider model (d_model 128 → 384) helped the most, a longer context did not help, and five blocks overfit. The full log is in [`experiments.md`](experiments.md).



## Limitations
 
- Each number comes from a single run. The GePpeTto runs use a fixed seed (42); the character-level runs do not.
- The transformer and GPT notebooks split the text by position (90/5/5), so validation and test come from the last part of the Paradiso. The MLP uses a different tokenization (words) and a split by whole cantos, so its loss is not comparable with theirs.
- The sample prompts are lines from the training text.



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
    - canto headers and blank separator lines are dropped, since they can pollute the training data.
    - the data is split by whole cantos (random within each cantica: 79 train / 9 val / 12 test) instead of shuffling context-target windows, because overlapping windows from adjacent text would leak near-duplicate examples across splits. Within a split the cantos are concatenated, so a few context windows cross canto boundaries.
    - the output is saved to `data/dante_preprocessed.pt`.

- **`mlp.ipynb`**: a first baseline, a word-level MLP with BatchNorm (vocabulary of 12,002 words built from the training cantos; 8% of the validation tokens are `<UNK>`), built without `nn.Module`, so every matrix multiplication, embedding lookup and BatchNorm update is explicit. This was to build intuition about activation scales, initialization, and why naive weight init blows up logits. Validation loss is 6.31 and test loss 6.34 per word (uniform guessing: 9.39). The generated text is not linguistically coherent.
- **`trf.ipynb`**: a character-level Transformer (multi-head self-attention, residual connections, LayerNorm, feed-forward blocks, all written by hand), trained on the cased text with a 90/5/5 split by position. It produces Dante-like text in style and line layout, though grammar and rhyme are still inconsistent.
- **`trf_v2.ipynb`**: the same architecture using `nn.MultiheadAttention`, used for the later runs on Colab (batch 64, more blocks, longer context, lower learning rate).
- **`GPT.ipynb`**: GePpeTto on the *Commedia*: zero-shot baseline, full fine-tuning and LoRA, with qualitative sampling for each.
- **`geppetto-dante-lora/`**: the trained LoRA adapter (~1 MB).
- **`experiments.md`**: hyperparameters and outcomes for the transformer and GePpeTto runs.


The text comes from [dmf.unicatt.it/~della/pythoncourse18/commedia.txt](https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt). In the transformer and GPT notebooks the canto headers and title lines are removed (551,846 → 534,889 characters).


## Setup
 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
`mlp.ipynb` needs `data/dante_preprocessed.pt`, which is not in the repo: run `EDA.ipynb` first to create it.

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
- Compare full fine-tuning and LoRA with the same training budget.
- RNN and LSTM baselines.


## References 
- MLP-based language modeling: [Bengio et al.](https://www.alphaxiv.org/38e694ba-c6f9-490d-a699-cf0aae33db13)
- Italian GPT fine-tuning: [De Mattei et al.](https://www.alphaxiv.org/pdf/2004.14253v1)
- LoRA: [Hu et al.](https://www.alphaxiv.org/pdf/2106.09685)
- Structure of the from-scratch models follows Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.
