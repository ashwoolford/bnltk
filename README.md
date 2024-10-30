# BNLTK
[![Build Status](https://travis-ci.org/ashwoolford/bnltk.svg?branch=master)](https://travis-ci.org/ashwoolford/bnltk)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/bnltk)](https://pepy.tech/project/bnltk)



BNLTK(Bangla Natural Language Processing Toolkit) is a open-source python package for Bengali Natural Language Processing. It includes modules for Tokenization, Stemming, Parts of speech tagging.

## installation

```
pip install bnltk 
```

## Usage

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

### Parts of Tagger

To use the Parts of Speech Tagger, please download the pretrained model's weights. Our trained model achieves an accuracy of 96%
```
from bnltk.bnltk_downloads import DataFiles
DataFiles().download()	
```
After successfully downloading the files, you can use this module as follows:

```
from bnltk.pos_tagger import PosTagger

p_tagger = PosTagger()
print(p_tagger.tagger('দুশ্চিন্তার কোন কারণই নাই'))  
# [('দুশ্চিন্তার', 'NC'), ('কোন', 'JQ'), ('কারণই', 'NC'), ('নাই', 'VM')]
```

Description of the POS tag set

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


