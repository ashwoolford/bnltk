# Bangla Natural Language Toolkit: Stemmer
# Rules: Rafi kamal  
# Copyright (C) 2019 BNLTK Project
# Author: Ashraf Hossain <asrafhossain197@gmail.com>

import string
import re


class StaticArrays:
	rule_words = ['ই', 'ও', 'তো', 'কে', 'তে', 'রা', 'চ্ছি', 'চ্ছিল','চ্ছে', 'চ্ছিস','চ্ছিলেন', 'চ্ছ', 'য়েছে', 'েছ', 'েছে','েছেন', 'রছ', 'রব', 'েল', 'েলো', 'ওয়া', 'েয়ে', 'য়', 'য়ে', 'েয়েছিল', 'েছিল', 'য়েছিল','েয়েছিলেন','ে.েছিলেন', 'েছিলেন', 'লেন', 'দের', 'ে.ে', 'ের','ার','েন', 'বেন', 'িস', 'ছিস', 'ছিলি', 'ছি', 'ছে', 'লি', 'বি','ে', 'টি', 'টির', 'েরটা', 'েরটার', 'টা', 'টার', 'গুলো', 'গুলোর', 'েরগুলো', 'েরগুলোর']
	rule_dict = {"রছ":"র","রব":"র","েয়ে":"া","েয়েছিল":"া","েয়েছিলেন":"া","ে.েছিলেন":"া.","ে.ে":"া."}        
	

class BanglaStemmer:

	def __init__(self):
		pass
	
	def repetition_checker(self, word, lin):
		return word == lin

	def len_checker(self, temp_arr):
		index = None
		word = None
		current_len = -1

		for i in range(0, len(temp_arr)):
			if len(temp_arr[i]) > current_len:
				current_len = len(temp_arr[i])
				index = i
				word = temp_arr[index]
		return word  

	def stem(self, lin):

	    temp_arr = []
	    flag = 0

	    for word in StaticArrays.rule_words:
	        if re.search('.*' + word + '$' ,lin):
	            temp_arr.append(word)

	    if len(temp_arr) != 0:
	        longest_word = self.len_checker(temp_arr)
	        if StaticArrays.rule_dict.get(longest_word):

	        	sliced = lin.replace(longest_word, StaticArrays.rule_dict[longest_word])
	        	if self.repetition_checker(sliced, lin):
	        		return lin[0] + 'া' + lin[2] + 'া'
	        	else:
	        		return sliced
	        else:
	            new_index = len(lin) - len(longest_word)
	            return lin[0:new_index]
	    else:
	        return lin

#print(BanglaStemmer().stem('তুমিও'))


