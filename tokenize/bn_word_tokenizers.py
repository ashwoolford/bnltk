# Bangla Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

import string 
import re
from string import punctuation

class Tokenizers:
	def __init__(self):
		pass

	def bn_word_tokenizer(self, input_):
		r = re.compile(r'[\s\।{}]+'.format(re.escape(punctuation)))
		list_ = r.split(input_)
		list_ = [i for i in list_ if i] 
		return list_