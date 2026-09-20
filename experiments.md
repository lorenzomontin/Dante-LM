# Tracking 
Here I keep track of all the tuning and run on the transformer architecture.

Experiments (char-level transformer): 
 
| Exp | Corpus | Context | d_model | Heads | Blocks | Val loss | 
| --- | ------ | ------: | ------: | ----: | -----: | -------: | 
| 0   | raw    |      64 |     128 |     4 |      1 |     1.81 | 
| 1   | clean  |      64 |     128 |     4 |      1 |     1.84 | 
| 2   | clean  |     128 |     128 |     4 |      1 |     1.89 | 
| 3   | clean  |     256 |     128 |     4 |      1 |     1.99 | 
| 4   | clean  |     128 |     384 |     4 |      1 |     1.60 | 
| 5   | clean  |     128 |     384 |     4 |      2 |     1.52 | 
| 6   | clean  |     128 |     384 |     4 |      2 |  pending | 
 
Exp 0–5: hand-written attention (`trf.ipynb`). Exp 6: same config with `nn.MultiheadAttention` and batch 64 (`trf_v2.ipynb`), result not recorded yet.



## Experiment 0 — baseline
 
- Character-level
- vocab = 69
- block_size = 64
- d_model = 128
- n_heads = 4
- d_head = 32
- d_ff = 512
- n_blocks = 1
- steps = 3000
Final validation loss ≈ 1.81
 
**Outcome:** qualitatively Dante-like, but poor grammar/rhyme,contains unwanted numerals/artifacts. 
 
 
## Experiment 1 — editorial lines removed
 
- Character-level
- vocab = 67
- block_size = 64
- d_model = 128
- n_heads = 4
- d_head = 32
- d_ff = 512
- n_blocks = 1
- steps = 3000
- corpus: canto headers and title lines stripped (551,846 → 534,889 characters)
Final validation loss ≈ 1.84
 
**Outcome:** qualitatively Dante-like, but poor grammar/rhyme,contains unwanted numerals/artifacts. 
 
 
## Experiment 2 — larger context window
 
- Character-level
- vocab = 67
- block_size = 128
- d_model = 128
- n_heads = 4
- d_head = 32
- d_ff = 512
- n_blocks = 1
- steps = 3000
Final validation loss ≈ 1.89
 
**Outcome:** poor grammar/rhyme, larger context window does not yield better understanding. 
 
## Experiment 3 — even larger context window
 
- Character-level
- vocab = 67
- block_size = 256
- d_model = 128
- n_heads = 4
- d_head = 32
- d_ff = 512
- n_blocks = 1
- steps = 3000
Final validation loss ≈ 1.99
 
**Outcome:** poor grammar/rhyme, larger context window does not yield better understanding: val loss gets worse as context grows at fixed capacity (1.84 → 1.89 → 1.99 for 64 → 128 → 256). 
 
## Experiment 4 —  larger embedding dimension (standard size for small models)
 
- Character-level
- vocab = 67
- block_size = 128
- d_model = 384
- n_heads = 4
- d_head = 96
- d_ff = 1536
- n_blocks = 1
- steps = 3000
Final validation loss ≈ 1.60
 
**Outcome:** poor grammar/rhyme, best performance achieved so far but training takes time. 
 
## Experiment 5 —  as before but two transformer block tested
 
- Character-level
- vocab = 67
- block_size = 128
- d_model = 384
- n_heads = 4
- d_head = 96
- d_ff = 1536
- n_blocks = 2
- batch_size = 32
- learning_rate = 3e-4 (AdamW)
- steps = 3000 (eval every 300 steps, 50 batches)
Final validation loss ≈ 1.52 (last logged, step 2700: train 1.45 / val 1.52)
 
Val loss trajectory: 1.56 (step 2100) → 1.54 (2400) → 1.52 (2700), still decreasing; the train/val gap is starting to open (~0.07).
 
**Outcome:** poor grammar/rhyme, best performance achieved so far but training takes considerable time. Sample (500 chars, temperature 1.0): Italian-looking words and Dante-like line layout, but many non-words and no rhyme:
 
    a dico, per procede a quel foco suono
    dispetto mi a cantova la terra;
      io sì dice sua dolce la cordesta
 
# Experiment 6 — same model with PyTorch `nn.MultiheadAttention` (`trf_v2.ipynb`)
 
- Character-level, clean corpus
- vocab = 67
- block_size = 128
- d_model = 384
- n_heads = 4
- d_head = 96 (implicit)
- d_ff = 1536
- n_blocks = 2
- batch_size = 64 (Exp 5 used 32)
- learning_rate = 3e-4 (AdamW)
- steps = 3000 (eval every 300 steps, 50 batches)
- attention: `nn.MultiheadAttention(batch_first=True)` with a causal mask, replacing the hand-written Head / MultiHeadAttention; pre-LN blocks otherwise identical
- built for Colab GPU runs
Final validation loss: not recorded yet (the uploaded notebook has no training output; last run was on CPU)



**Outcome:** pending. Note that batch size differs from Exp 5, so a difference in val loss is not only due to the attention implementation.
 

---

## GePpeTto fine-tuning tracking  

Common setup: GePpeTto (Italian GPT-2, ~109M params), BPE tokenizer (vocab = 30000), same cleaned Commedia text and 90/5/5 sequential train/val/test split as the transformer, block_size = 128. Losses are cross-entropy per BPE token (HF `labels=`, shifted internally), perplexity = exp(loss). They are not comparable with the per-character losses above.


| Exp | Method | Trainable params | LR | Batch | Steps | Val loss | Test loss (ppl) |
| --- | ------ | ---------------: | -: | ----: | ----: | -------: | --------------: |
| ref | zero-shot, no training | 0 | – | 4 | – | 6.32 | – |
| 0   | full fine-tuning (no LoRA) | ~108.9M (all) | 5e-5 | 4 | 100 | 4.20 | 4.20 (66.4) |
| 1   | LoRA r=8, α=16, `c_attn` | 294,912 (0.27%) | 2e-4 | 8 | 1000 | 4.14 | not computed yet |

### Reference - zero-shot (before any training)
 
- Single-batch loss check: 6.04 (perplexity 419.8), 1 batch of 4 × 128, so noisy
- 20-batch estimate (from the LoRA sanity check, where the untrained adapter must equal the base model): train 6.12 / val 6.32 (perplexity ~554 on val)
- Prompt "Nel mezzo del cammin di nostra vita": the top next-token is "," (p ≈ 0.34)
**Outcome:** fluent modern Italian, no Dante. Sampled continuation drifts into contemporary prose ("...Maria ha potuto fruire di un faticoso alleggerimento della memoria della madre...").


### Experiment 0 - baseline (no LoRA): full fine-tuning
 
- BPE tokenization (GePpeTto GPT-2 tokenizer, vocab = 30000)
- block_size = 128
- batch_size = 4
- learning_rate = 5e-5 (AdamW)
- max_steps = 100 (~51k tokens seen)
- all weights trained (~108.9M)
- eval every 10 steps, 10 batches
Last logged (step 90): train 4.06 / val 4.20
Final test loss ≈ 4.196 (20 batches)
Final test perplexity ≈ 66.4
 
Superseded earlier record: test loss ~5.90 / perplexity ~367.60 with gibberish output ("di, chel nonò'r, mète ',"). The notebook now reproduces 4.196 / 66.4 instead.
 
**Outcome:** coherent Italian again, with archaic Dante-like forms ('l, ne la, 'ntesi), but repetitive ("mente", "mondo" recur constantly) and no terza rima or rhyme. Val estimate is noisy (bounces between 4.19 and 4.28 over steps 40–90).
 
### Experiment 1 - LoRA
 
- Same base model, tokenizer, data and block_size = 128
- r = 8, lora_alpha = 16 (scale alpha/r = 2), lora_dropout = 0.05
- target_modules = ["c_attn"] (fused Q/K/V in each of the 12 blocks), fan_in_fan_out = True (GPT-2 Conv1D)
- trainable params = 294,912 of 109,177,344 (0.27%)
- batch_size = 8
- learning_rate = 2e-4 (AdamW, LoRA params only)
- max_steps = 1000 (~1.02M tokens seen)
- eval every 100 steps, 20 batches
- adapter saved to ./geppetto-dante-lora
Sanity check before training (B = 0): train 6.12 / val 6.32, identical to the zero-shot model.
Last logged (step 900): train 4.04 / val 4.14 (val perplexity ~63)
Val trajectory: 4.50 (step 100) → 4.32 → 4.27 → 4.25 → 4.22 → 4.22 → 4.21 → 4.17 → 4.14 (step 900), still decreasing.
Test loss / perplexity: not computed yet (the final-metrics cell was run before the LoRA section).
 
**Outcome:** qualitatively similar to Exp 0 on 3 prompts (1 sample each): archaic lexicon and register (sanza, 'l, ne lo cielo, li occhi), dialogue with quotation marks, but still repetitive ("la tua mente e con la tua mente"), a stray non-Latin character in one sample, irregular line lengths and no consistent rhyme scheme. Val 4.14 vs 4.20 is not a like-for-like win for LoRA: it saw ~20× more tokens (1.02M vs 51k) with a different batch size and learning rate.


