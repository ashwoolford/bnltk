# BNLTK
[![Build Status](https://travis-ci.org/ashwoolford/bnltk.svg?branch=master)](https://travis-ci.org/ashwoolford/bnltk)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)


BNLTK(Bangla Natural Language Processing Toolkit) is open-source python package for Bengali Natural Language Processing. It includes modules for Tokenization, Stemming, Parts of speech tagging. I'm looking forward to helping form contributors to make this look far better than this.

## installation

pip install bnltk 

## Usage

### Tokenizer

```
from bnltk.tokenize import Tokenizers
t = Tokenizers()
print(t.bn_word_tokenizer(' আমার সোনার বাংলা । '))		
```

### Stemmer

```
from bnltk.stemmer import BanglaStemmer
bn_stemmer = BanglaStemmer()
print(bn_stemmer.stem('খেয়েছিলো'))
```

### Parts of Tagger

For using the Parts of Tagger you need to download some data files as follows:

```
from bnltk.bnltk_downloads import DataFiles
DataFiles().download()	
```
After successfully downloading the files, then you can use this module.

```
pos_tagger = PosTagger()    
pos_tagger.loader()
sentences = 'দুশ্চিন্তার কোন কারণই নাই'
print(pos_tagger.tagger(sentences))  

```
