import unittest
from bnltk.stemmer import BanglaStemmer


class TestStemmer(unittest.TestCase):

    def setUp(self):
        self.t = BanglaStemmer()

    def tearDown(self):
        pass

    def test_stem_is_defined(self):
        self.assertTrue(callable(self.t.stem))

    def test_bn_word_tokenizer_should_not_raise_error(self):
        self.assertEqual(self.t.stem(), "")
        self.assertEqual(self.t.stem(""), "")
        self.assertEqual(self.t.stem([]), "")
        self.assertEqual(self.t.stem({}), "")

    def test_stem(self):
        self.assertEqual(self.t.stem("রহিমের"), "রহিম")
        self.assertEqual(self.t.stem("হেসেছিলেন"), "হাসা")
        self.assertEqual(self.t.stem("রহিমেরটার"), "রহিম")
        self.assertEqual(self.t.stem("বাচ্চারা"), "বাচ্চা")


if __name__ == "__main__":
    unittest.main()
