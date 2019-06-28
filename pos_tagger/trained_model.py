# Bangla Natural Language Toolkit: Parts of Speech Tagger
#
# Copyright (C) 2019 BNLTK Project
# Author: Ashraf Hossain <asrafhossain197@gmail.com>


from keras.models import load_model
from string import punctuation
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import platform
import getpass
import os
import sys

import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Loader:

	texts = ''
	sentences = []
	model = ''

	model_path = None
	tagged_data_path = None

	def __init__(self):
		self.texts = ''
		self.sentences = []
		self.model = None
		self.model_path = None
		self.tagged_data_path = None

	def path_generator(self):

		isFiles_exist = True

		if platform.system() == 'Windows':
		    self.model_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\keras_mlp_bangla.h5"
		    self.tagged_data_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\bn_tagged_mod.txt"
		else:
		    self.model_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/keras_mlp_bangla.h5"
		    self.tagged_data_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/bn_tagged_mod.txt" 
			

	def load_keras_model(self):

		self.path_generator()

		self.model = load_model(self.model_path)
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.load_corpus()
		self.data_manipulator()

	def load_corpus(self):
		#file = '/Users/ashrafhossain/bnltk_data/pos_data/bn_tagged_mod.txt'
		self.texts = open(self.tagged_data_path, encoding="utf8").readlines()

	def tuple_maker(self, line):
	    line = line.split(' ')
	    sentence = []
	    
	    for x in line:
	        
	        if x == '':
	            continue
	        else:
	            x = x.split('\\')
	            tup = []
	            for y in x:
	                tup.append(y);
	            sentence.append(tuple(tup))
	    return sentence            

	def data_manipulator(self):
		for i in self.texts:
			self.sentences.append(self.tuple_maker(i))


class BanglaPosTagger:

	sentences = []
	mod_elements = []
	model = ''
	dict_vectorizer = None
	label_encoder = None


	def __init__(self):
		self.sentences = []
		self.mod_elements = []
		self.model = ''
		self.dict_vectorizer = DictVectorizer(sparse=False)
		self.label_encoder = LabelEncoder()

	def load(self):

		loader_ = Loader()
		loader_.load_keras_model()
		self.model = loader_.model
		self.sentences = loader_.sentences
		#print(self.sentences[0])
		#print(self.mod_elements)


		train_test_cutoff = int(.80 * len(self.sentences))
		training_sentences = self.sentences[:train_test_cutoff]
		testing_sentences = self.sentences[train_test_cutoff:]
		train_val_cutoff = int(.25 * len(training_sentences))
		validation_sentences = training_sentences[:train_val_cutoff]
		training_sentences = training_sentences[train_val_cutoff:]

		X_train, y_train = self.transform_to_dataset(training_sentences)
		X_test, y_test = self.transform_to_dataset(testing_sentences)
		X_val, y_val = self.transform_to_dataset(validation_sentences)

		#dict_vectorizer = DictVectorizer(sparse=False)
		self.dict_vectorizer.fit(X_train + X_test + X_val)
		self.label_encoder.fit(y_train + y_test + y_val)	
			

	def bn_pos_tag(self, input):

		self.load()

		self.bn_tokenizer(input)

		t_list = self.training_transform_to_dataset([self.mod_elements])
		t_list = self.dict_vectorizer.transform(t_list)
		#print(t_list)

		predictions = self.model.predict(t_list)

		list_ = []
		for x in range(0, len(predictions)):
			list_.append(np.argmax(predictions[x]))
		#label_encoder = LabelEncoder()


		labels = self.label_encoder.inverse_transform(list_)

		result = []

		for i in range(0, len(labels)):
			tup = []
			tup.append(self.mod_elements[i])
			tup.append(labels[i])
			result.append(tuple(tup))

		return result	


	def bn_tokenizer(self, input_):


		words = input_.split(' ')
		words = [x.strip(' ') for x in words] 
		words = [i for i in words if i] 

		dict_ = {}
		dict_['।'] = True

		for p in punctuation:
			dict_[p] = True

		for n in words:
			if dict_.get(n[-1]):
				self.mod_elements.append(n[:-1])
				self.mod_elements.append(n[-1])
			else:
				self.mod_elements.append(n)
		self.mod_elements = [i for i in self.mod_elements if i]

	def add_basic_features(self, sentence_terms, index):
    
	    #print(sentence_terms[index])
	    """ Compute some very basic word features.

	        :param sentence_terms: [w1, w2, ...] 
	        :type sentence_terms: list
	        :param index: the index of the word 
	        :type index: int
	        :return: dict containing features
	        :rtype: dict
	    """
	    term = sentence_terms[index]
	    return {
	        'nb_terms': len(sentence_terms),
	        'term': term,
	        'is_first': index == 0,
	        'is_last': index == len(sentence_terms) - 1,
	        'prefix-1': term[0],
	        'prefix-2': term[:2],
	        'prefix-3': term[:3],
	        'suffix-1': term[-1],
	        'suffix-2': term[-2:],
	        'suffix-3': term[-3:],
	        'prev_word': '' if index == 0 else sentence_terms[index - 1],
	        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
	    }

	def training_transform_to_dataset(self, tagged_sentences):
	    """
	    Split tagged sentences to X and y datasets and append some basic features.

	    :param tagged_sentences: a list of POS tagged sentences
	    :param tagged_sentences: list of list of tuples (term_i, tag_i)
	    :return: 
	    """
	    X = []
	    
	    #print(len(tagged_sentences))

	    for pos_tags in tagged_sentences:
	        #print(pos_tags)
	        for index in range(len(pos_tags)):
	            # Add basic NLP features for each sentence term
	            X.append(self.add_basic_features(pos_tags, index))
	    return X

	def untag(self, tagged_sentence):
	    """ 
	    Remove the tag for each tagged term. 

	    :param tagged_sentence: a POS tagged sentence
	    :type tagged_sentence: list
	    :return: a list of tags
	    :rtype: list of strings
	    """
	    return [w for w, _ in tagged_sentence]

	def transform_to_dataset(self, tagged_sentences):
	    
	    """
	    Split tagged sentences to X and y datasets and append some basic features.

	    :param tagged_sentences: a list of POS tagged sentences
	    :param tagged_sentences: list of list of tuples (term_i, tag_i)
	    :return: 
	    """
	    X, y = [], []

	    for pos_tags in tagged_sentences:
	        for index, (term, class_) in enumerate(pos_tags):
	            # Add basic NLP features for each sentence term
	            X.append(self.add_basic_features(self.untag(pos_tags), index))
	            y.append(class_)
	    return X, y
'''
t = BanglaPosTagger()
t.load()
print(t.bn_pos_tag(' আমার সোনার বাংলা । আমি তোমায় ভালোবাসি । '))	
''' 			

