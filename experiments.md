# Experiments
 
Log of the Dante LM experiments: a character-level transformer trained from scratch (`trf.ipynb`, `trf_v2.ipynb`) and a fine-tuned Italian GPT-2 (GePpeTto, `GPT.ipynb`). Only the runs that changed a conclusion are kept.
 
---
 
## 1. Character-level transformer
 
**Common setup:** clean corpus (canto headers and titles stripped, 551,846 → 534,889 characters), vocab = 67, n_heads = 4, d_head = d_model / 4, d_ff = 4 × d_model, AdamW, single seed. Batch size and learning rate were not recorded for runs 1–4. Runs 5–8 come from `trf_v2.ipynb` (`nn.MultiheadAttention`, built for Colab). Losses are per character, at the last step.
 
| Run | Old # | Context | d_model | Blocks | LR | Steps | Batch | Train | Val |
| --- | ----: | ------: | ------: | -----: | -: | ----: | ----: | ----: | --: |
| 1   | 1     |      64 |     128 |      1 | –    | 3000 | – | –    | 1.84 |
| 2   | 2     |     128 |     128 |      1 | –    | 3000 | – | –    | 1.89 |
| 3   | 3     |     256 |     128 |      1 | –    | 3000 | – | –    | 1.99 |
| 4   | 4     |     128 |     384 |      1 | –    | 3000 | – | –    | 1.60 |
| 5   | 6     |     128 |     384 |      2 | 3e-4 | 3000 | 64 | 1.34 | 1.49 |
| 6   | 7     |     128 |     384 |      5 | 3e-4 | 3000 | 64 | 0.92 | 1.73 |
| 7   | 9     |     128 |     384 |      3 | 1e-4 | 3000 | 64 | 1.54 | 1.57 |
| 8   | 11    |     512 |     384 |      3 | 1e-4 | 5000 | 64 | 1.45 | 1.51 |
 
### Findings
 
- **Context length (runs 1–3):** at d_model = 128, a longer context does not help. Val loss gets worse as it grows (1.84 → 1.89 → 1.99 for 64 → 128 → 256).
- **Width (run 4):** d_model 128 → 384 is the largest single gain (1.89 → 1.60 at context 128).
- **Depth (runs 5–7):** 2 blocks is best (1.49). 5 blocks overfits clearly (train 0.92 vs val 1.73), and 4 blocks behaves the same. 3 blocks at lr 1e-4 does not overfit (gap ~0.03) but learns more slowly (1.57).
- **Long context, low lr (run 8):** context 512 at lr 1e-4 is underfit at 3000 steps (val 1.68) and reaches 1.51 at 5000 steps, still falling.
- **Overall:** best val loss sits around 1.5 across quite different configurations. Samples are Dante-like in style and line layout, but ungrammatical and unrhymed.
Run 5 has the same config as an earlier hand-written-attention run (batch 32, val 1.52 at step 2700). The difference is too small to attribute to the implementation, since batch size and final step also differ.
 
**Not listed:** raw-corpus baseline (old Exp 0, val 1.81, not comparable: vocab 69, headers with numerals); hand-written 2-block run (old Exp 5, see above); context 256 with 3 blocks (old Exp 8, train 1.19 / val 1.51, changed blocks and context together); context 512 at 3000 steps (old Exp 10, folded into run 8).
 
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
 
**Common setup:** GePpeTto (Italian GPT-2, ~109M params), BPE vocab = 30000, same cleaned text and 90/5/5 sequential train/val/test split, block_size = 128, AdamW. Losses are per BPE token (perplexity = exp(loss)), so they are not comparable with the per-character losses above.
 
| Run | Method | Trainable params | LR | Batch | Steps | Tokens seen | Val loss | Test loss (ppl) |
| --- | ------ | ---------------: | -: | ----: | ----: | ----------: | -------: | --------------: |
| Zero-shot | none | 0 | – | 4 | – | – | 6.32 | – |
| Full fine-tune | all weights | ~108.9M | 5e-5 | 4 | 100 | ~51k | 4.20 | 4.20 (66.4) |
| LoRA | r = 8, α = 16, dropout 0.05, `c_attn` | 294,912 (0.27%) | 2e-4 | 8 | 1000 | ~1.02M | 4.14 | not computed yet |
 
Val loss is the last logged value (step 90 and step 900).
 
### Findings
 
- **Zero-shot:** fluent modern Italian, no Dante. After "Nel mezzo del cammin di nostra vita" the top next token is "," (p ≈ 0.34), and sampled continuations drift into contemporary prose.
- **Full fine-tune:** coherent Italian with archaic Dante-like forms ('l, ne la, 'ntesi), but repetitive ("mente", "mondo" recur constantly) and no terza rima. The val estimate is noisy (4.19–4.28 over steps 40–90, from only 10 batches of 4).
- **LoRA:** the untrained adapter reproduces the base model's loss (train 6.12 / val 6.32), as expected. Val falls from 4.50 (step 100) to 4.14 (step 900) and is still decreasing. Qualitatively similar to the full fine-tune on 3 prompts (archaic lexicon, dialogue quotes) with the same repetition, a stray non-Latin character in one sample, irregular line lengths and no consistent rhyme scheme.
- **Caveat:** LoRA's val 4.14 vs 4.20 is not a like-for-like win. It saw ~20× more tokens with a different batch size and learning rate. LoRA test metrics are not computed yet.
