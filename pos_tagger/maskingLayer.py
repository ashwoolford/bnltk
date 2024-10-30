# Bangla Natural Language Toolkit: Masking Layer
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

import tensorflow as tf
from tensorflow.keras.layers import Layer


class MaskingLayer(Layer):
    def call(self, inputs):
        # Create an attention mask to ignore padding tokens (0s in inputs)
        return tf.cast(tf.math.not_equal(inputs, 0), tf.float32)[
            :, tf.newaxis, tf.newaxis, :
        ]
