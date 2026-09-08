# Dante- LM

This projects explore the poem 'Divine Comedy' by Dante Alighieri, the father of Italian language. The poem, considered one of the greatest works of Western literature, depicts Dante's travels through Hell, Purgatory, and Heaven. Allegorically, the poem represents the soul's journey towards God. Similarly, this projects aims to be a journey into Deep Learning, exploring different architectures to understand their mechanisms.


Model explored and tested:
- MLP  https://www.alphaxiv.org/38e694ba-c6f9-490d-a699-cf0aae33db13
- Small transformer 
- Fine Tuning a GPT https://www.alphaxiv.org/pdf/2004.14253v1
- RNN (later)
- LSTM (later)



# Structure 
Notebooks:

- **EDA.ipnyb**: exploratory data analysis with tokenization and data splitting.
    Preprocessing decisions:
    - lowercase text (at least for now) for simplicity.
    - words separated by apostrophe(') are kept together, e.g. ch'io.
    - punctuation symbols are treated separately, to gather further information on Dante's writing style.   
    - editorial parts dropped: headlines with roman numerals, publisher's notes, and blank separation lines. All of these can 'pollute' the training data. 
    - Cantos structure is kept: canto boundaries are recorded as token-index markers (not baked into the token stream itself). This is needed for two things: 
    1) splitting train/val/test by whole cantos rather than shuffling individual context-target windows, since overlapping windows from adjacent canto text would otherwise leak near-duplicate examples across the split; 
    
    2) preventing any single sliding window from straddling a canto seam, even after splitting by canto.

- **MLP.ipynb**: first NN baseline, MLP with BatchNorm. The network directly uses the data coming from the EDA notebook. Although the loss looks promising, the output is gibberish.

- **Trf.ipynb**: Transformer trained and implemented. The outputs shows better results and further refinements are carried out.