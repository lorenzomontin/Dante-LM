# Tracking 
Here I keep track of all the tuning and run on the transformer architecture.

Experiments planned: 

| Exp | Corpus | Context | d_model | Heads | Blocks | Val loss | 
| --- | ------ | ------: | ------: | ----: | -----: | -------: | 
| 0   | raw    |      64 |     128 |     4 |      1 |     1.81 | 
| 1   | clean  |      64 |     128 |     4 |      1 |     1.84 | 
| 2   | clean  |     128 |     128 |     4 |      1 |     1.89 | 
| 3   | clean  |     256 |     128 |     4 |      1 |     1.99 | 
| 4   | clean  |     128 |     384 |     4 |      1 |     1.60 | 
| 5   | clean  |     128 |     384 |     4 |      2 |     1.52 | 




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

Final validation loss ≈ 1.84

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

Final validation loss ≈ 1.84

**Outcome:** poor grammar/rhyme, larger context window does not yield better understanding. 

## Experiment 4 —  larger context window with standard embedding space for small models

- Character-level
- vocab = 67
- block_size = 128
- d_model = 384
- n_heads = 4
- d_head = 32
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
- d_head = 32
- d_ff = 1536
- n_blocks = 2
- steps = 3000

Final validation loss ≈ 1.52

**Outcome:** poor grammar/rhyme, best performance achieved so far but training takes considerable time. 



---

## GePpeTto fine-tuning tracking  

### Experiment 0 - baseline (no LoRA)

- BPE tokenization (GePpeTto GPT-2 tokenizer, vocab = 30000)
- block_size = 128
- batch_size = 4
- learning_rate = 5e-5
- max_steps = 100

Final test loss ~ 5.90
Final test perplexity ~ 367.60

**Outcome:** Baseline, no rhymes and gibberish Italian: 
Nel mezzo del cammin di nostra vita
 di, chel nonò'r, mète ',
'' di' suor e' a' suoi'