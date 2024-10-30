# Bangla Natural Language Toolkit: Transformer Encoder Layer
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Layer


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(dff, activation="relu"),  # Pointwise feed-forward network
                Dense(d_model),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
