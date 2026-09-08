# 


### EDA
Preprocessing decisions:
- lowercase text (at least for now) for simplicity.
- words separated by apostrophe(') are kept together, e.g. ch'io.
- punctuation symbols are treated separately, to gather further information on Dante's writing style.   
- editorial parts dropped: headlines with roman numerals, publisher's notes, and blank separation lines. All of these can 'pollute' the training data. 
- Cantos structure is kept: canto boundaries are recorded as token-index markers (not baked into the token stream itself). This is needed for two things: 
 1) splitting train/val/test by whole cantos rather than shuffling individual context-target windows, since overlapping windows from adjacent canto text would otherwise leak near-duplicate examples across the split; 
 
2) preventing any single sliding window from straddling a canto seam, even after splitting by canto.



## Standard NLP drill

RAW TEXT
   ↓
inspect
   ↓
clean / normalize
   ↓
tokenize
   ↓
create vocabulary
   ↓
encode tokens → integers
   ↓
create context → target pairs
   ↓
train/validation/test split
   ↓
MLP


# Project
The projets want to assess whether LMs can reproduce the style of Dante Alighieri, the father of Italian language. Like Dante's epic journey through different realms, this projects explore diffrent LMs architecture. 

Model explored and tested:
- MLP  https://www.alphaxiv.org/private/38e694ba-c6f9-490d-a699-cf0aae33db13
- RNN
- LSTM
- Small transformer (optionally, with more data)


# Structure 
Notebooks:

- **EDA.ipnyb**: exploratory data analysis with tokenization and data splitting.
- **MLP.ipynb**: first NN baseline, MLP with BatchNorm. The network learns but the output is gibberish.
- **Trf.ipynb**: Transformer trained and implemented. The outputs shows promising resutls and further refinements are carried out.