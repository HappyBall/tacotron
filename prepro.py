# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
from hyperparams import Hyperparams as hp
import sys

if not os.path.exists(hp.prepro_path): os.mkdir(hp.prepro_path)
mel_path = os.path.join(hp.prepro_path, "mels")
mag_path = os.path.join(hp.prepro_path, "mags")

# Load data
fpaths, _, _ = load_data() # list

if not os.path.exists(mel_path): os.mkdir(mel_path)
if not os.path.exists(mag_path): os.mkdir(mag_path)

for fpath in tqdm.tqdm(fpaths):
    try:
        fname, mel, mag = load_spectrograms(fpath)
    except:
        print(fpath + "\n")
        sys.stdout.flush()
        continue
    np.save(mel_path + "/{}".format(fname.replace("wav", "npy")), mel)
    np.save(mag_path + "/{}".format(fname.replace("wav", "npy")), mag)
