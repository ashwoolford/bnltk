# BNLTK

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/bnltk)](https://pepy.tech/project/bnltk)



BNLTK(Bangla Natural Language Processing Toolkit) is an open-source python package for Natural Language Processing in Bangla. It offers functionalities to perform some basic NLP tasks such as Tokenization, Stemming and Parts of speech tagging. BNLTK requires Python version 3.6, 3.7, 3.8, 3.9 or 3.10.

Web documentation: [https://ashwoolford.github.io/bnltk/](https://ashwoolford.github.io/bnltk/)

## installation

```
pip install bnltk 
```

**Note**: If you are using version 0.7.6, please see the documentation [here](#version-076)


## Version 0.7.8 (latest)

### Tokenizer

```
from bnltk.tokenize import Tokenizers
t = Tokenizers()
print(t.bn_word_tokenizer('আজ আবহাওয়া খুব ভালো।'))
# ["আজ", "আবহাওয়া", "খুব", "ভালো", "।"]
```

### Stemmer

```
from bnltk.stemmer import BanglaStemmer
bn_stemmer = BanglaStemmer()
print(bn_stemmer.stem('হেসেছিলেন'))
# হাসা
```

### Parts of speech tagger

To use the Parts of Speech Tagger, please download the pretrained model's weights. Our trained model achieves an accuracy of 96%
```
from bnltk.bnltk_downloads import DataFiles
DataFiles.download()	
```
After successfully downloading the files, you can use this module as follows:

```
from bnltk.pos_tagger import PosTagger

p_tagger = PosTagger()
print(p_tagger.tagger('দুশ্চিন্তার কোন কারণই নাই'))  
# [('দুশ্চিন্তার', 'NC'), ('কোন', 'JQ'), ('কারণই', 'NC'), ('নাই', 'VM')]
```

## Version 0.7.6

### Tokenizer

```
from bnltk.tokenize import Tokenizers
t = Tokenizers()
print(t.bn_word_tokenizer('আজ আবহাওয়া খুব ভালো।'))
# ["আজ", "আবহাওয়া", "খুব", "ভালো"]
```

### Stemmer

```
from bnltk.stemmer import BanglaStemmer
bn_stemmer = BanglaStemmer()
print(bn_stemmer.stem('হেসেছিলেন'))
# হাসা
```

### Parts of speech tagger

To use the Parts of Speech Tagger, please download the pretrained model's weights. Our trained model achieves an accuracy of 96%
```
from bnltk.bnltk_downloads import DataFiles
DataFiles().download()	
```
After successfully downloading the files, you can use this module as follows:

```
from bnltk.pos_tagger import PosTagger

p_tagger = PosTagger()
p_tagger.loader()
print(p_tagger.tagger('দুশ্চিন্তার কোন কারণই নাই'))  
# [('দুশ্চিন্তার', 'NC'), ('কোন', 'JQ'), ('কারণই', 'NC'), ('নাই', 'VM')]
```

### Description of the POS tag set

| Categories            | Types                 |
|-----------------------|-----------------------|
| Noun (N)              | Common (NC)           |
|                       | Proper (NP)           |
|                       | Verbal (NV)           |
|                       | Spatio-temporal (NST) |
| Pronoun (P)           | Pronominal (PPR)      |
|                       | Reflexive (PRF)       |
|                       | Reciprocal (PRC)      |
|                       | Relative (PRL)        |
|                       | Wh (PWH)              |
|                       |                       |
| Nominal Modifier (J)  | Adjectives (JJ)       |
|                       | Quantifiers (JQ)      |
| Demonstratives (D)    | Absolutive (DAB)      |
|                       | Relative (DRL)        |
|                       | Wh (DWH)              |
| Adverb (A)            | Manner (AMN)          |
|                       | Location (ALC)        |
| Participle (L)        | Relative (LRL)        |
|                       | Verbal (LV)           |
| Postposition (PP)     |                       |
| Particles (C)         | Coordinating (CCD)    |
|                       | Subordinating (CSB)   |
|                       | Classifier (CCL)      |
|                       | Interjection (CIN)    |
|                       | Others (CX)           |
| Punctuations (PU)     |                       |
| Residual (RD)         | Foreign Word (RDF)    |
|                       | Symbol (RDS)          |
|                       | Other (RDX)           |


