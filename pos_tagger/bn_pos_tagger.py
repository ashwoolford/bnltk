# Bangla Natural Language Toolkit: Parts of Speech Tagger
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import platform
import getpass

from bnltk.pos_tagger.maskingLayer import MaskingLayer
from bnltk.pos_tagger.positionalEncoding import PositionalEncoding
from bnltk.pos_tagger.transformerEncoderLayer import TransformerEncoderLayer
from bnltk.tokenize.bn_word_tokenizers import Tokenizers


# Parameters
VOCAB_SIZE = 10000  # Vocabulary size
MAX_SEQ_LEN = 250  # Longest sequence
D_MODEL = 256  # Dimension of embeddings
NUM_HEADS = 4  # Number of attention heads
DFF = 256  # Feed-forward dimension in the Transformer
NUM_LAYERS = 2  # Number of transformer encoder layers
DROPOUT_RATE = 0.2  # Dropout rate
NUM_TAGS = 33  # Number of pos tags


class PosTagger:

    def __init__(self):
        platform_sys = platform.system()

        if platform_sys == "Windows":
            self.saved_weight_path = (
                "C:\\Users\\"
                + getpass.getuser()
                + "\\bnltk_data\\pos_data\\pos_tagger.weights.h5"
            )
            self.corpus_data_path = (
                "C:\\Users\\"
                + getpass.getuser()
                + "\\bnltk_data\\pos_data\\bn_tagged_mod.txt"
            )
        elif platform_sys == "Linux":
            self.saved_weight_path = (
                "/home/"
                + getpass.getuser()
                + "/bnltk_data/pos_data/pos_tagger.weights.h5"
            )
            self.corpus_data_path = (
                "/home/" + getpass.getuser() + "/bnltk_data/pos_data/bn_tagged_mod.txt"
            )
        elif platform_sys == "Darwin":
            self.saved_weight_path = (
                "/Users/"
                + getpass.getuser()
                + "/bnltk_data/pos_data/pos_tagger.weights.h5"
            )
            self.corpus_data_path = (
                "/Users/" + getpass.getuser() + "/bnltk_data/pos_data/bn_tagged_mod.txt"
            )
        else:
            raise Exception("Unable to detect OS")

        self.model = self._build_pos_tagger(
            VOCAB_SIZE,
            MAX_SEQ_LEN,
            NUM_TAGS,
            D_MODEL,
            NUM_HEADS,
            DFF,
            NUM_LAYERS,
            DROPOUT_RATE,
        )
        self.model.load_weights(self.saved_weight_path)

        # corpus_data = open(self.corpus_data_path, encoding="utf8").readlines()
        with open(self.corpus_data_path, encoding="utf8") as file:
            corpus_data = file.readlines()
        corpus_data = [self._tuple_maker(sentence) for sentence in corpus_data]

        # Extract words and tags
        self.sentences = [[word for word, _ in sentence] for sentence in corpus_data]
        self.tags = [[tag for _, tag in sentence] for sentence in corpus_data]

        # Create a vocabulary for words and POS tags
        self.word_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=VOCAB_SIZE, oov_token="<OOV>"
        )
        self.word_tokenizer.fit_on_texts(self.sentences)

        self.tag_encoder = LabelEncoder()
        flat_tags = [tag for sent_tags in self.tags for tag in sent_tags]
        self.tag_encoder.fit(flat_tags)

    # Build the Transformer POS Tagger Model
    def _build_pos_tagger(
        self,
        vocab_size,
        max_seq_len,
        num_tags,
        d_model=64,
        num_heads=2,
        dff=128,
        num_layers=2,
        dropout_rate=0.1,
    ):
        inputs = Input(shape=(max_seq_len,))
        mask = MaskingLayer()(inputs)  # Create the mask using MaskingLayer

        # Embedding Layer + Positional Encoding
        embedding = Embedding(vocab_size, d_model)(inputs)
        positional_encoding = PositionalEncoding(max_seq_len, d_model)(embedding)
        x = positional_encoding

        # Transformer Encoder Layers
        for _ in range(num_layers):
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(
                x, training=True, mask=mask
            )

        # Final Dense Layer to predict POS tags
        outputs = Dense(num_tags, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def loader(self):
        warnings.warn("loader() is deprecated now, it has no affect")

    def _tuple_maker(self, sentence):
        elements = sentence.split()

        result = []

        for element in elements:
            if element != "":
                text, tag = element.rsplit("\\", 1)
                result.append((text, tag))

        return result

    def tagger(self, input_str=""):

        if not isinstance(input_str, str):
            warnings.warn(
                "tagger() expected a string as arg, but got a non-string value."
            )
            return []

        if input_str.strip() == "":
            warnings.warn("tagger() expected a not empty string as arg")
            return []

        words = Tokenizers.bn_word_tokenizer(input_str)

        tokenized_sentence = self.word_tokenizer.texts_to_sequences([words])
        tokenized_sentence = pad_sequences(
            tokenized_sentence, maxlen=MAX_SEQ_LEN, padding="post"
        )

        predictions = self.model.predict(tokenized_sentence, verbose=0)
        predicted_tag_indices = np.argmax(predictions, axis=-1)
        predicted_tags = [
            self.tag_encoder.inverse_transform([index])[0]
            for index in predicted_tag_indices[0][: len(words)]
        ]

        tagged_results = [
            (word, tag.item()) for word, tag in zip(words, predicted_tags)
        ]

        return tagged_results
