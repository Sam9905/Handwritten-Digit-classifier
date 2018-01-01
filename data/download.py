#!/usr/bin/env python
"""
RUN THIS CELL TO DOWNLOAD THE MNIST DATA SET
"""
from urllib.request import urlretrieve
from os import access, path, R_OK
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import gzip
import glob

FILE_NAME = 'mnist.pkl.gz'
FILE_PATH = './data'
FILE_REMOTE_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
     
    
class DSProgress(object):
    
    def test_folder_path(self, file_path):
        assert file_path is not None,\
        'MNIST data file not set.'
        print('All files found!')
        
    def setup_data_files(self):
        if not isfile(FILE_NAME):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='MNIST Dataset') as pbar:
                 urlretrieve(FILE_REMOTE_URL ,FILE_NAME, pbar.hook)

        if not isdir(FILE_NAME):
            with gzip.open(FILE_NAME, 'rb') as in_file:
                s = in_file.read()

            # Now store the uncompressed data
            FILE_TO_STORE = FILE_NAME[:-3]  # remove the '.gz' from the filename
            # store uncompressed file data from 's' variable
            with open(FILE_TO_STORE, 'wb') as f:
                f.write(s)
            self.test_folder_path(FILE_TO_STORE)
        else:
            print ("Either file is missing or is not readable")

