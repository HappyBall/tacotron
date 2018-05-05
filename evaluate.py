# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from graph import Graph
from utils import spectrogram2wav
from scipy.io.wavfile import write
import os
import numpy as np
from utils import load_spectrograms

def mse(list1, list2):
    return ((list1-list2) ** 2).mean(axis=None)

def calculate_mse(arr1, arr2):
    if len(arr1) > len(arr2):
        result = np.zeros(arr1.shape)
        result[:arr2.shape[0]] = arr2
        return mse(arr1, result)
    else:
        result = np.zeros(arr2.shape)
        result[:arr1.shape[0]] = arr1
        return mse(arr2, result)


def evaluate():
    # Load graph
    g = Graph(mode="evaluate"); print("Graph loaded")

    # Load data
    fpaths, _, texts = load_data(mode="evaluate")
    lengths = [len(t) for t in texts]
    maxlen = sorted(lengths, reverse=True)[0]
    new_texts = np.zeros((len(texts), maxlen), np.int32)
    for i, text in enumerate(texts):
        new_texts[i, :len(text)] = [idx for idx in text]
    #new_texts = np.split(new_texts, 2)
    new_texts = new_texts[:300]
    half_size = int(len(fpaths)/2)
    print(half_size)
    #new_fpaths = [fpaths[:half_size], fpaths[half_size:]]
    fpaths = fpaths[:300]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Evaluate Model Restored!")
        """
        err = 0.0

        for i, t_split in enumerate(new_texts):
            y_hat = np.zeros((t_split.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
            for j in tqdm.tqdm(range(200)):
                _y_hat = sess.run(g.y_hat, {g.x: t_split, g.y: y_hat})
                y_hat[:, j, :] = _y_hat[:, j, :]

            mags = sess.run(g.z_hat, {g.y_hat: y_hat})
            for k, mag in enumerate(mags):
                fname, mel_ans, mag_ans = load_spectrograms(new_fpaths[i][k])
                print("File {} is being evaluated ...".format(fname))
                audio = spectrogram2wav(mag)
                audio_ans = spectrogram2wav(mag_ans)
                err += calculate_mse(audio, audio_ans)

        err = err/float(len(fpaths))
        print(err)

        """
        # Feed Forward
        ## mel
        y_hat = np.zeros((new_texts.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
        for j in tqdm.tqdm(range(200)):
            _y_hat = sess.run(g.y_hat, {g.x: new_texts, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]
        ## mag
        mags = sess.run(g.z_hat, {g.y_hat: y_hat})
        err = 0.0
        for i, mag in enumerate(mags):
            fname, mel_ans, mag_ans = load_spectrograms(fpaths[i])
            print("File {} is being evaluated ...".format(fname))
            audio = spectrogram2wav(mag)
            audio_ans = spectrogram2wav(mag_ans)
            err += calculate_mse(audio, audio_ans)
        err = err/float(len(fpaths))
        print(err)


if __name__ == '__main__':
    evaluate()
    print("Done")

