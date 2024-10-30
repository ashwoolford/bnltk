# Bangla Natural Language Toolkit: DataFilles Downloader
#
# Copyright (C) 2019-2024 BNLTK Project
# Author: Asraf Patoary <asrafhossain197@gmail.com>

from requests import get
import platform
import getpass
import os
import sys


class DataFiles:
    @staticmethod
    def _downloader(url, file_name, tag):
        if not os.path.exists(file_name):
            with open(file_name, "wb") as file:
                print("Downloading....../" + tag)
                response = get(url, stream=True)
                total_length = response.headers.get("content-length")

                if total_length is None:
                    file.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        file.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                        sys.stdout.flush()
        else:
            print(tag + "is already exists!!")
            
    @staticmethod
    def download():
        file_name = None
        tag1 = "bn_tagged_mod.txt"
        tag2 = "pos_tagger.weights.h5"

        if platform.system() == "Windows":
            file_name = "C:\\Users\\" + getpass.getuser()
        elif platform.system() == "Linux":
            file_name = "/home/" + getpass.getuser()
        elif platform.system() == "Darwin":
            file_name = "/Users/" + getpass.getuser()
        else:
            raise Exception("Unable to detect OS")

        corpus_url = "https://firebasestorage.googleapis.com/v0/b/diu-question.appspot.com/o/nlp_data%2Fbn_tagged_mod.txt?alt=media&token=00f383a3-f913-480b-85c1-971dd8fd6dd9"
        saved_weights_url = "https://firebasestorage.googleapis.com/v0/b/diu-question.appspot.com/o/nlp_data%2Fpos_tagger.weights.h5?alt=media&token=2251eedd-dfaf-4572-9bce-b4d293cce980"

        try:
            os.makedirs(file_name + "/bnltk_data/pos_data")
        except OSError:
            print("Creation of the directory failed or exists")

        DataFiles._downloader(
            corpus_url, file_name + "/bnltk_data/pos_data/bn_tagged_mod.txt", tag1
        )
        DataFiles._downloader(
            saved_weights_url,
            file_name + "/bnltk_data/pos_data/pos_tagger.weights.h5",
            tag2,
        )
        print("Done!")