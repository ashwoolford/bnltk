# Bangla Natural Language Toolkit: Parts of Speech Tagger
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>



import string
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout, Input, Layer
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

import platform
import getpass

import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from maskingLayer import MaskingLayer
from positionalEncoding import PositionalEncoding
from transformerEncoderLayer import TransformerEncoderLayer


# Parameters
VOCAB_SIZE = 10000    # Vocabulary size
MAX_SEQ_LEN = 250     # Longest sequence
D_MODEL = 256         # Dimension of embeddings
NUM_HEADS = 4         # Number of attention heads
DFF = 256             # Feed-forward dimension in the Transformer
NUM_LAYERS = 2        # Number of transformer encoder layers
DROPOUT_RATE = 0.2    # Dropout rate
NUM_TAGS = 33         # Number of pos tags


class PosTagger:
    
    def __init__(self):
        platform_sys = platform.system()

        if platform_sys == 'Windows':
            self.saved_weight_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\pos_tagger.weights.h5"
            self.corpus_data_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\bn_tagged_mod.txt"
        elif platform_sys == 'Linux':
            self.saved_weight_path = "/home/"+getpass.getuser()+"/bnltk_data/pos_data/pos_tagger.weights.h5"
            self.corpus_data_path = "/home/"+getpass.getuser()+"/bnltk_data/pos_data/bn_tagged_mod.txt" 
        elif platform_sys == 'Darwin':
            self.saved_weight_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/pos_tagger.weights.h5"
            self.corpus_data_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/bn_tagged_mod.txt" 
        else:
            raise Exception('Unable to detect OS')
        
        self.model = self.build_pos_tagger(VOCAB_SIZE, MAX_SEQ_LEN, NUM_TAGS, D_MODEL, NUM_HEADS, DFF, NUM_LAYERS, DROPOUT_RATE)
        self.model.load_weights(self.saved_weight_path)

        corpus_data = open(self.corpus_data_path, encoding='utf8').readlines()
        corpus_data = [self.tuple_maker(sentence) for sentence in corpus_data]

        # Extract words and tags
        self.sentences = [[word for word, _ in sentence] for sentence in corpus_data]
        self.tags = [[tag for _, tag in sentence] for sentence in corpus_data]

        # Create a vocabulary for words and POS tags
        self.word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        self.word_tokenizer.fit_on_texts(self.sentences)

        self.tag_encoder = LabelEncoder()
        flat_tags = [tag for sent_tags in self.tags for tag in sent_tags]
        self.tag_encoder.fit(flat_tags)
        


    # Build the Transformer POS Tagger Model
    def build_pos_tagger(self, vocab_size, max_seq_len, num_tags, d_model=64, num_heads=2, dff=128, num_layers=2, dropout_rate=0.1):
        inputs = Input(shape=(max_seq_len,))
        mask = MaskingLayer()(inputs)  # Create the mask using MaskingLayer

        # Embedding Layer + Positional Encoding
        embedding = Embedding(vocab_size, d_model)(inputs)
        positional_encoding = PositionalEncoding(max_seq_len, d_model)(embedding)
        x = positional_encoding

        print('diff ', dff)

        # Transformer Encoder Layers
        for _ in range(num_layers):    
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x, training=True, mask=mask)

        # Final Dense Layer to predict POS tags
        outputs = Dense(num_tags, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    
    dict_vectorizer = None
    label_encoder = None
    model = None

    def loader(self):
        pass
        # global dict_vectorizer
        # global label_encoder
        # global model

        # self.saved_weight_path = None
        # self.corpus_data_path = None

        # if platform.system() == 'Windows':
        #     self.saved_weight_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\pos_tagger.weights.h5"
        #     self.corpus_data_path = "C:\\Users\\"+getpass.getuser()+"\\bnltk_data\\pos_data\\bn_tagged_mod.txt"
        # elif platform.system() == 'Linux':
        #     self.saved_weight_path = "/home/"+getpass.getuser()+"/bnltk_data/pos_data/pos_tagger.weights.h5"
        #     self.corpus_data_path = "/home/"+getpass.getuser()+"/bnltk_data/pos_data/bn_tagged_mod.txt" 
        # elif platform.system() == 'Darwin':
        #     self.saved_weight_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/pos_tagger.weights.h5"
        #     self.corpus_data_path = "/Users/"+getpass.getuser()+"/bnltk_data/pos_data/bn_tagged_mod.txt" 
        # else:
        #     raise Exception('Unable to detect OS')
        
        # model = keras.saving.load_model(self.saved_weight_path)
        # # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(
        #     loss=keras.losses.BinaryCrossentropy(),
        #     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        #     metrics=[
        #         keras.metrics.BinaryAccuracy()
        #     ]
        # )

        # #new_file = 'bn_tagged_mod.txt'
        # texts = open(self.corpus_data_path, encoding='utf8').readlines()
        # sentences = []
        # for i in texts:
        #     sentences.append(self.tuple_maker(i))

        # #print(sentences[0])


        # train_test_cutoff = int(.80 * len(sentences)) 
        # training_sentences = sentences[:train_test_cutoff]
        # testing_sentences = sentences[train_test_cutoff:]

        # train_val_cutoff = int(.25 * len(training_sentences))
        # validation_sentences = training_sentences[:train_val_cutoff]
        # training_sentences = training_sentences[train_val_cutoff:]

        # X_train, y_train = self.transform_to_dataset(training_sentences)
        # X_test, y_test = self.transform_to_dataset(testing_sentences)
        # X_val, y_val = self.transform_to_dataset(validation_sentences)

        # dict_vectorizer = DictVectorizer(sparse=False)
        # dict_vectorizer.fit(X_train + X_test + X_val)

        # label_encoder = LabelEncoder()
        # label_encoder.fit(y_train + y_test + y_val)

    def tuple_maker(self, line):
        sentence = []
        line = line.split(' ')

        for x in line:

            if x == '':
                print("Yess")
            else:
                x = x.split('\\')
                tup = []
                for y in x:
                    tup.append(y);
                sentence.append(tuple(tup))    

        return sentence

    def tokenizer(self, input_):

        mod_elements = []

        words = input_.split(' ')
        words = [x.strip(' ') for x in words] 
        words = [i for i in words if i] 

        dict_ = {}
        dict_['।'] = True

        for p in string.punctuation:
            dict_[p] = True

        for n in words:
            if dict_.get(n[-1]):
                mod_elements.append(n[:-1])
                mod_elements.append(n[-1])
            else:
                mod_elements.append(n)
        mod_elements = [i for i in mod_elements if i]
        return mod_elements     

    def add_basic_features(self, sentence_terms, index):

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
        X = []

        #print(len(tagged_sentences))

        for pos_tags in tagged_sentences:
            #print(pos_tags)
            for index in range(len(pos_tags)):
                # Add basic NLP features for each sentence term
                X.append(self.add_basic_features(pos_tags, index))
        return X

    def untag(self, tagged_sentence):
        return [w for w, _ in tagged_sentence]

    def transform_to_dataset(self, tagged_sentences):
        X, y = [], []

        for pos_tags in tagged_sentences:
            for index, (term, class_) in enumerate(pos_tags):
                # Add basic NLP features for each sentence term
                X.append(self.add_basic_features(self.untag(pos_tags), index))
                y.append(class_)
        return X, y

    def tagger(self, sentences):

        #elements = sentences.split(' ')
        mod_elements = self.tokenizer(sentences)
        print("mod_elements ", mod_elements)
        #print(mod_elements)
        t_list = self.training_transform_to_dataset([mod_elements])
        print('t_list ', t_list)
        # t_list = dict_vectorizer.transform(t_list)

        # predictions = model.predict(t_list)
        # list_ = []
        # for x in range(0, len(predictions)):
        #     list_.append(np.argmax(predictions[x]))
        # list_  = label_encoder.inverse_transform(list_)  

        return []

tt = PosTagger()    
# tt.loader()
# sentences = 'দুশ্চিন্তার কোন কারণই নাই'
# print(tt.tagger(sentences))      

# texts = open('/home/apa/bnltk_data/pos_data/bn_tagged_mod.txt', encoding='utf8').readlines()
# sentences = []
# for i in texts:
#     sentences.append(tt.tuple_maker(i))

# print(sentences[:5])


# print(tt.tuple_maker('রপ্তানি\JJ দ্রব্য\NC -\PU তাজা\JJ ও\CCD শুকনা\JJ ফল\NC ,\PU আফিম\NC ,\PU পশুচর্ম\NC ও\CCD পশম\NC এবং\CCD কার্পেট\NC ৷\PU'))