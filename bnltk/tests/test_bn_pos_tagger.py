import unittest
from bnltk.pos_tagger import PosTagger


class TestPosTagger(unittest.TestCase):

    def setUp(self):
        self.pos_tagger = PosTagger()

    def tearDown(self):
        pass

    def test_tagger_is_defined(self):
        self.assertTrue(callable(self.pos_tagger.tagger))

    def test_tagger_should_not_raise_error(self):
        self.assertEqual(self.pos_tagger.tagger(""), [])
        self.assertEqual(self.pos_tagger.tagger("   "), [])
        self.assertEqual(self.pos_tagger.tagger(123), [])
        self.assertEqual(self.pos_tagger.tagger([]), [])
        self.assertEqual(self.pos_tagger.tagger({}), [])

    def test_tagger(self):
        self.assertNotEqual(self.pos_tagger.tagger("তুমি কোথায় যাচ্ছ?"), [])
        self.assertNotEqual(self.pos_tagger.tagger("আমি বই পড়ছি।"), [])
        self.assertNotEqual(self.pos_tagger.tagger("বাচ্চারা খেলছে।"), [])


if __name__ == "__main__":
    unittest.main()
