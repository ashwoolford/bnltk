# Bangla Natural Language Toolkit: DataFilles Downloader
#
# Copyright (C) 2019 BNLTK Project
# Author: Ashraf Hossain <asrafhossain197@gmail.com>

from requests import get  # to make GET request
import platform
import getpass
import os
import sys


class DataFiles:
	def __init__(self):
		pass

	def downloader(self, url, file_name, tag):
		if not os.path.exists(file_name):
				    # open in binary mode
		    with open(file_name, "wb") as file:
		        # get request
		        print("Downloading....../"+tag)
		        response = get(url, stream=True)
		        # write to file
		        #file.write(response.content)
		        
		        
		        total_length = response.headers.get('content-length')

		        if total_length is None: # no content length header
		            file.write(response.content)
		        else:
		            dl = 0
		            total_length = int(total_length)
		            for data in response.iter_content(chunk_size=4096):
		                dl += len(data)
		                file.write(data)
		                done = int(50 * dl / total_length)
		                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
		                sys.stdout.flush()
		else:
			print(tag + 'is already exists!!')           	

		     

	def download(self):
		file_name = None
		tag1 = 'bn_tagged_mod.txt'
		tag2 = 'keras_mlp_bangla.h5'

		if platform.system() == 'Windows':
		    file_name = "C:\\Users\\"+getpass.getuser()
		else:
		    file_name = "/Users/"+getpass.getuser()
		#print(file_name)
		url = 'https://firebasestorage.googleapis.com/v0/b/diu-question.appspot.com/o/nlp_data%2Fbn_tagged_mod.txt?alt=media&token=00f383a3-f913-480b-85c1-971dd8fd6dd9'
		url2 = 'https://firebasestorage.googleapis.com/v0/b/diu-question.appspot.com/o/nlp_data%2Fkeras_mlp_bangla.h5?alt=media&token=4146c1b0-1e4d-4f9e-8b2f-7e3519106a40'


		try:  
		    os.makedirs(file_name+'/bnltk_data/pos_data')
		except OSError:  
		    print ("Creation of the directory failed or exists")
		else:  
		    pass   

		self.downloader(url, file_name+'/bnltk_data/pos_data/bn_tagged_mod.txt', tag1) 
		print()
		self.downloader(url2, file_name+'/bnltk_data/pos_data/keras_mlp_bangla.h5', tag2) 
		print('Done!')

