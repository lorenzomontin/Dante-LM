# Experiments
 
Log of the Dante LM experiments: a character-level transformer trained from scratch (`trf.ipynb`, `trf_v2.ipynb`) and a fine-tuned Italian GPT-2 (GePpeTto, `GPT.ipynb`, `ablation/lora_ablation.ipynb`). Only the runs that changed a conclusion are kept.
 
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
 
| Run | Method | Trainable params | LR | Batch | Steps | Val loss | Test loss (ppl) |
| --- | ------ | ---------------: | -: | ----: | ----: | -------: | --------------: |
| Zero-shot | none | 0 | – | – | – | 6.30 | 6.31 (552.6) |
| Full fine-tune (`GPT.ipynb`) | all weights | ~108.9M | 5e-5 | 4 | 100 | 4.17 | 4.19 (66.0) |
| LoRA (`GPT.ipynb`) | r=8, α=16, `c_attn` | 294,912 (0.27%) | 2e-4 | 8 | 1000 | 4.15 | 4.13 (61.9) |


### 2.1 LoRA ablation (`ablation/lora_ablation.ipynb`, seed 42, single run each)
 
One-factor-at-a-time from the reference config (r=8, `c_attn`, lr 2e-4, batch 8, 1000 steps). Full fine-tune re-run at 500 steps (see 2.2 for why).
 
| Run | Method | Targets | Rank | LR | Steps | Trainable | Val loss (ppl) |
| --- | ------ | ------- | ---: | -: | ----: | --------: | --------------: |
| lora_r8_ref | LoRA | `c_attn` | 8 | 2e-4 | 1000 | 294,912 (0.27%) | 4.14 (62.6) |
| lora_r4 | LoRA | `c_attn` | 4 | 2e-4 | 1000 | 147,456 (0.14%) | 4.17 (64.7) |
| lora_r16 | LoRA | `c_attn` | 16 | 2e-4 | 1000 | 589,824 (0.54%) | 4.09 (59.6) |
| lora_r8_all_linear | LoRA | `c_attn`+`c_proj`+`c_fc` | 8 | 2e-4 | 1000 | 1,179,648 (1.07%) | 3.95 (51.9) |
| lora_r8_lr1e-4 | LoRA | `c_attn` | 8 | 1e-4 | 1000 | 294,912 (0.27%) | 4.20 (66.5) |
| lora_r8_lr5e-4 | LoRA | `c_attn` | 8 | 5e-4 | 1000 | 294,912 (0.27%) | 4.06 (57.7) |
| full_ft_lr5e-5 | full | all weights | – | 5e-5 | 500 | 108,882,432 (100%) | 4.00 (54.7), best at step 300–450, see 2.2 |
 
**Findings:** target modules matter most (extending from `c_attn` alone to all three linear layers: 4.14 → 3.95, the single largest change); rank has a small, monotonic effect (r=4 → 8 → 16: 4.17 → 4.14 → 4.09); learning rate is also monotonic in this range (1e-4 → 2e-4 → 5e-4: 4.20 → 4.14 → 4.06, not yet showing instability). `lora_r8_all_linear` is the best single-seed config and was taken forward to the matched comparison below.


### 2.2 Full fine-tune overfits quickly; matched comparison (3 seeds each)
 
The full fine-tune's train loss falls monotonically to 2.33 at step 1000 while its val loss bottoms out around step 300–450 (~3.94–3.95) and then rises again (4.37 by step 1000) — classic overfitting on a ~156k-token train split. Reporting its step-1000 val loss (as in the original `GPT.ipynb` run and the first ablation pass) understates it. It was re-run at 500 steps with `eval_interval=50` to catch the actual best point, and both it and the best LoRA config (`lora_r8_all_linear`) were then run on 3 seeds (1, 2, 3):
 
| Config | Val loss (mean ± std) | Test loss (mean ± std) | Wall time |
| --- | ---: | ---: | ---: |
| Full fine-tune, 500 steps, best-val | 3.987 ± 0.011 | 3.976 ± 0.004 | 2.40 min |
| LoRA r=8, all-linear, 1000 steps | 3.957 ± 0.010 | 3.958 ± 0.009 | 3.56 min |
 
**Finding:** at matched best-val, LoRA and full fine-tuning are statistically indistinguishable (Δ≈0.03, smaller than either method's own run-to-run std of ~0.01). LoRA reaches this training 1.08% of the parameters, at no risk of the overfitting seen in the unconstrained full fine-tune. The original headline gap (LoRA 4.13 vs full fine-tune 4.19 test loss) was mostly an artifact of comparing 1000 LoRA steps against an under-trained 100-step full fine-tune, and the reverse gap seen with a fixed step-1000 comparison (this section) was an artifact of the full fine-tune's overfitting, not evaluated with early stopping.
 
### Qualitative findings (all runs)
 
- **Zero-shot:** fluent modern Italian, no Dante. After "Nel mezzo del cammin di nostra vita" the top next token is "," (p ≈ 0.34), and sampled continuations drift into contemporary prose (Roman history, in the sample). One stray Arabic-script character already appears in this zero-shot sample.
- **Full fine-tune and LoRA (`c_attn` only, `GPT.ipynb`):** coherent Italian with archaic Dante-like forms (*sanza*, *'l*, *ne la*, *Ed elli a me*), but repetitive ("non vedi" / "di più che di più che di più" recur) and no terza rima. A stray non-Latin character appears in one LoRA sample, but the base model produces it too.
- **LoRA, all-linear (best config, local re-run, seed 1, val 3.955 at step 1000):** clearly the most fluent and Dante-like output of any run. Correct terzina indentation, consistent archaic register (*sanza*, *'l*, *etterna*, *rispuose*), in-character dialogue tags ("io: '...'", "disse, '...'"), and grammatically well-formed Italian sentences throughout both samples: a step up from the repetitive, less coherent output of the `c_attn`-only run. Still no consistent end rhyme and it does not follow the ABA BCB terza rima scheme.


