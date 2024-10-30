import unittest
from bnltk.tokenize import Tokenizers


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bn_word_tokenizer_is_defined(self):
        self.assertTrue(callable(Tokenizers.bn_word_tokenizer))

    def test_bn_word_tokenizer_should_not_raise_error(self):
        self.assertEqual(Tokenizers.bn_word_tokenizer(""), [])
        self.assertEqual(Tokenizers.bn_word_tokenizer(123), [])
        self.assertEqual(Tokenizers.bn_word_tokenizer([]), [])
        self.assertEqual(Tokenizers.bn_word_tokenizer({}), [])

    def test_bn_word_tokenizer(self):
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("আমি বই পড়ছি।"), ["আমি", "বই", "পড়ছি", "।"]
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("তুমি কোথায় যাচ্ছ?"),
            ["তুমি", "কোথায়", "যাচ্ছ", "?"],
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("আজ আবহাওয়া খুব ভালো।"),
            ["আজ", "আবহাওয়া", "খুব", "ভালো", "।"],
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("বাচ্চারা,,, খেলছে।।"),
            ["বাচ্চারা", ",", ",", ",", "খেলছে", "।", "।"],
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("বাচ্চারা,, , খেলছে।।"),
            ["বাচ্চারা", ",", ",", ",", "খেলছে", "।", "।"],
        )
        self.assertEqual(Tokenizers.bn_word_tokenizer("বাচ্চারা !"), ["বাচ্চারা", "!"])
        self.assertEqual(Tokenizers.bn_word_tokenizer("বাচ্চারা!"), ["বাচ্চারা", "!"])
        self.assertEqual(Tokenizers.bn_word_tokenizer("বাচ্চারা"), ["বাচ্চারা"])
        self.assertEqual(Tokenizers.bn_word_tokenizer("বাচ্চারা?"), ["বাচ্চারা", "?"])
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("বাচ্চারা??"), ["বাচ্চারা", "?", "?"]
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("তুমি, কোথায় যাচ্ছ!"),
            ["তুমি", ",", "কোথায়", "যাচ্ছ", "!"],
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("তুমি, কোথায় যাচ্ছ, "),
            ["তুমি", ",", "কোথায়", "যাচ্ছ", ","],
        )
        self.assertEqual(
            Tokenizers.bn_word_tokenizer("        তুমি, কোথায়           যাচ্ছ,   "),
            ["তুমি", ",", "কোথায়", "যাচ্ছ", ","],
        )


if __name__ == "__main__":
    unittest.main()
