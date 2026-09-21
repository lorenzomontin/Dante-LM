# Experiments
 
Log of the Dante LM experiments: a character-level transformer trained from scratch (`trf.ipynb`, `trf_v2.ipynb`) and a fine-tuned Italian GPT-2 (GePpeTto, `GPT.ipynb`). Only the runs that changed a conclusion are kept.
 
---
 
## 1. Character-level transformer
 
**Common setup:** clean corpus (canto headers and titles stripped, 551,846 → 534,889 characters), vocab = 67, n_heads = 4, d_head = d_model / 4, d_ff = 4 × d_model, AdamW, single seed. Batch size and learning rate were not recorded for runs 1–4. Runs 5–8 come from `trf_v2.ipynb` (`nn.MultiheadAttention`, built for Colab). Losses are per character, at the last step.
 
| Run | Context | d_model | Blocks | LR | Steps | Batch | Train | Val |
| --- | ---: | ------: | -----: | -: | ----: | ----: | ----: | --: |
| 1   |   64 |     128 |      1 | –    | 3000 | – | –    | 1.84 |
| 2   |  128 |     128 |      1 | –    | 3000 | – | –    | 1.89 |
| 3   |  256 |     128 |      1 | –    | 3000 | – | –    | 1.99 |
| 4   |  128 |     384 |      1 | –    | 3000 | – | –    | 1.60 |
| 5   |  128 |     384 |      2 | 3e-4 | 3000 | 64 | 1.34 | 1.49 |
| 6   |  128 |     384 |      5 | 3e-4 | 3000 | 64 | 0.92 | 1.73 |
| 7   |  128 |     384 |      3 | 1e-4 | 3000 | 64 | 1.54 | 1.57 |
| 8   |  512 |     384 |      3 | 1e-4 | 5000 | 64 | 1.45 | 1.51 |
 
### Findings
 
- **Context length (runs 1–3):** at d_model = 128, a longer context does not help. Val loss gets worse as it grows (1.84 → 1.89 → 1.99 for 64 → 128 → 256).
- **Width (run 4):** d_model 128 → 384 is the largest single gain (1.89 → 1.60 at context 128).
- **Depth (runs 5–7):** 2 blocks is best (1.49). 5 blocks overfits clearly (train 0.92 vs val 1.73), and 4 blocks behaves the same. 3 blocks at lr 1e-4 does not overfit (gap ~0.03) but learns more slowly (1.57).
- **Long context, low lr (run 8):** context 512 at lr 1e-4 is underfit at 3000 steps (val 1.68) and reaches 1.51 at 5000 steps, still falling.
- **Overall:** best val loss sits around 1.5 across quite different configurations. Samples are Dante-like in style and line layout, but ungrammatical and unrhymed.
Run 5 has the same config as an earlier hand-written-attention run (batch 32, val 1.52 at step 2700). The difference is too small to attribute to the implementation, since batch size and final step also differ.
 

### Samples
 
Run 5 (2 blocks):
 
```
e cominciai: "Qual volume morpi",
come tu mi rinse a' mortalita vaso".
  Ed elli a masti Giovessi e d'alto",
corrise, di vostri s'è per del pio sono.
  E unù per lo Strinsi ritorda
```
 
Run 8 (context 512, 5000 steps):
 
```
la Lucipio che v'ere poco de l'altre sozza
orperché, or tai chi rispuosi leva.
  Posciavo mi furo lar di quel figate
lume da lor, per leggel due a frelle,
```
 
---
 
## 2. GePpeTto fine-tuning
 
**Common setup:** GePpeTto (Italian GPT-2, ~109M params), BPE vocab = 30000, same cleaned text and 90/5/5 sequential train/val/test split (val and test are the last 10% of the text), block_size = 128, AdamW, seed 42. Losses are per BPE token (perplexity = exp(loss)), so they are not comparable with the per-character losses above. Val and test losses are computed over the full split in non-overlapping 128-token windows, so they are deterministic. The train loss printed during training is a random-batch estimate.
 
| Run | Method | Trainable params | LR | Batch | Steps | Tokens seen | Val loss | Test loss (ppl) |
| --- | ------ | ---------------: | -: | ----: | ----: | ----------: | -------: | --------------: |
| Zero-shot | none | 0 | – | – | – | – | 6.30 | 6.31 (552.6) |
| Full fine-tune | all weights | ~108.9M | 5e-5 | 4 | 100 | ~51k | 4.17 | 4.19 (66.0) |
| LoRA | r = 8, α = 16, dropout 0.05, `c_attn` | 294,912 (0.27%) | 2e-4 | 8 | 1000 | ~1.02M | 4.15 | 4.13 (61.9) |
 
Val loss is the last logged value (step 90 and step 900).

### Findings
 
- **Zero-shot:** fluent modern Italian, no Dante. After "Nel mezzo del cammin di nostra vita" the top next token is "," (p ≈ 0.34), and sampled continuations drift into contemporary prose (Roman history, in the sample). One stray Arabic-script character already appears in this zero-shot sample.
- **Full fine-tune:** val falls smoothly from 5.24 (step 0) to 4.17 (step 90), with train at 4.08. Coherent Italian with archaic Dante-like forms (*sanza*, *'l*, *Ed elli a me*), but repetitive ("non vedi" recurs constantly) and no terza rima.
- **LoRA:** the untrained adapter reproduces the zero-shot val loss exactly (6.3006), as expected. Val falls from 4.51 (step 100) to 4.15 (step 900) and is nearly flat over the last 200 steps (4.149 at step 800, 4.147 at step 900), with train at 4.08. Qualitatively similar to the full fine-tune on 3 prompts (archaic lexicon, dialogue quotes) with the same repetition ("di più che di più che di più"), a stray non-Latin character in one sample (the base model produces it too), irregular line lengths and no consistent rhyme scheme.
- **Caveat:** LoRA's test loss is lower than the full fine-tune's (4.13 vs 4.19), but this is not a like-for-like comparison. LoRA saw ~20× more tokens, with a different batch size and learning rate.

