import unittest 
from stemmer_builder import BanglaStemmer 


class TestStemmer(unittest.TestCase):

	def setUp(self):
		print('Setup')
		self.t = BanglaStemmer()

	def tearDown(self):	
		print('Tear Down\n')

	def test_stem(self):
		self.assertEqual(self.t.stem('রহিমের'),'রহিম')
		self.assertEqual(self.t.stem('হেসেছিলেন'),'হাসা')
		self.assertEqual(self.t.stem('রহিমেরটার'),'রহিম')


if __name__ == '__main__':
	unittest.main()		