# Bangla Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

import re
import warnings


class Tokenizers:
    @staticmethod
    def bn_word_tokenizer(input=""):
        if not isinstance(input, str):
            warnings.warn(
                "bn_word_tokenizer() expected arg as a string, but got a non-string value."
            )
            return []

        pattern = r"[\u0980-\u09FF]+|[^\s]"  # [\u0980-\u09FF]+: Matches one or more Bengali characters.
        tokens = re.findall(pattern, input)

        return tokens
