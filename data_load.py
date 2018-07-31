# -*- coding: utf-8 -*-
from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import re
import os
import json
import unicodedata
from bopomofo import to_bopomofo
from pypinyin import pinyin, Style

def load_vocab():
    # word base
    if hp.input_mode == "word":
        transcript = os.path.join(hp.prepro_path, 'transcript_training.txt')
        lines = open(transcript, "r").readlines()
        text_dict = dict()
        remove_key = list()
        count = 0
        char2idx = dict()
        idx2char = dict()

        for line in lines:
            text = ''.join(line.split()[1:])

            english_check = re.search('[a-zA-Z]', text)
            if english_check:
                continue

            for word in text:
                if word not in text_dict:
                    text_dict[word] = 0
                text_dict[word] += 1

        for key in text_dict:
            if text_dict[key] <= 5:
                remove_key.append(key)

        for key in remove_key:
            del text_dict[key]

        for idx, char in enumerate(text_dict):
            char2idx[char] = idx+1
            idx2char[idx+1] = char
        #char2idx = {char:idx for idx, char in enumerate(text_dict)}
        #idx2char = {idx:char for idx, char in enumerate(text_dict)}
        char2idx['P'] = 0
        idx2char[0] = 'P'
        length = len(char2idx)
        char2idx['oov'] = length
        char2idx['E'] = length+1
        idx2char[length] = 'oov'
        idx2char[length+1] = 'E'

    # bopomofo base
    elif hp.input_mode == "bopomofo" or hp.input_mode == "pinyin":
        char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
        idx2char = {idx: char for idx, char in enumerate(hp.vocab)}

    # phoneme base
    elif hp.input_mode == "phoneme":
        char2idx = json.load(open("./phoneme_preprocess/phone2idx.json", "r"))
        idx2char = json.load(open("./phoneme_preprocess/idx2phone.json", "r"))
    #syllable base
    else:
        transcript = os.path.join(hp.prepro_path, 'transcript_training.txt')
        lines = open(transcript, "r").readlines()
        syl_dict = dict()
        remove_key = list()
        count = 0
        char2idx = dict()
        idx2char = dict()

        for line in lines:
            text = ''.join(line.split()[1:])

            english_check = re.search('[a-zA-Z]', text)
            if english_check:
                continue

            if hp.input_mode == "syllable":
                # bopomofo syllable
                if hp.withtone:
                    text = to_bopomofo(text)
                else:
                    text = to_bopomofo(text, tones=False)
            else:
                # pinyin syllable
                text_pinyin = pinyin(text, style=Style.TONE3)
                text = ""
                for t in text_pinyin:
                    if hp.withtone:
                        text = text + " " + t[0]
                    else:
                        tmp = ''.join([i for i in t[0] if not i.isdigit()])
                        text = text + " " + tmp

            text = text.split()

            for syl in text:
                if syl not in syl_dict:
                    syl_dict[syl] = 0
                syl_dict[syl] += 1

        for key in syl_dict:
            if syl_dict[key] <= 5:
                remove_key.append(key)

        for key in remove_key:
            del syl_dict[key]

        char2idx = {char:idx for idx, char in enumerate(syl_dict)}
        idx2char = {idx:char for idx, char in enumerate(syl_dict)}

        """
        for idx, char in enumerate(syl_dict):
            char2idx[char] = idx+1
            idx2char[idx+1] = char
        char2idx['P'] = 0
        idx2char[0] = 'P'
        """
        length = len(char2idx)
        char2idx['oov'] = length
        char2idx['E'] = length+1
        idx2char[length] = 'oov'
        idx2char[length+1] = 'E'

    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    # Load vocabulary
    char2idx, idx2char = load_vocab()
    if hp.input_mode == "phoneme":
        word2phonelist = json.load(open("./phoneme_preprocess/word2phonelist.json", "r"))

    if mode in ("train", "eval", "evaluate"):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(hp.prepro_path, 'transcript_training.txt')
        if mode=="evaluate":
            transcript = os.path.join(hp.prepro_path, 'transcript_testing.txt')
        lines = open(transcript, 'r').readlines()
        total_hours = 0

        for line in lines:
            check_text = "".join(line.split()[1:])
            english_check = re.search('[a-zA-Z]', check_text)
            if english_check:
                continue

            #print(line)
            fname, text = line.strip().split()
            #text = text_normalize(text)

            if len(text) > hp.max_len:
                continue

            # bopomofo base
            if hp.input_mode == "pinyin" or hp.input_mode == "pinyin_syl":
                text_pinyin = pinyin(text, style=Style.TONE3)
                text = ""
                for t in text_pinyin:
                    if hp.withtone:
                        if hp.input_mode == "pinyin":
                            text = text + t[0]
                        else:
                            text = text + " " + t[0]
                    else:
                        tmp = ''.join([i for i in t[0] if not i.isdigit()])
                        if hp.input_mode == "pinyin":
                            text = text + tmp
                        else:
                            text = text + " " + t[0]
            elif hp.input_mode == "phoneme":
                t_tmp = ""
                for t in text:
                    if t not in word2phonelist:
                        t_tmp = t_tmp + " " + "oov"
                    else:
                        phonelist = word2phonelist[t]
                        t_tmp = t_tmp + " " + " ".join(phonelist)
                text = t_tmp
            elif hp.input_mode != "word":
                if hp.withtone:
                    text = to_bopomofo(text)
                else:
                    text = to_bopomofo(text, tones=False)

            if hp.input_mode != "pinyin" and hp.input_mode != "pinyin_syl" and hp.input_mode != "phoneme":
                text = text.replace("er", u"\u3126")
                text = text.replace("an", u"\u3122")
                text = text.replace("jue", u"\u3110\u3129\u311d")
                text = text.replace("xue", u"\u3112\u3129\u311d")
                text = text.replace("aE", u"\u311a")

                english_check = re.search('[a-zA-Z]', text)
                if english_check:
                    continue

            if hp.input_mode == "phoneme":
                text = text + " S"
            else:
                text = text + " E"

            temp = []

            if hp.input_mode != "syllable" and hp.input_mode != "pinyin_syl" and hp.input_mode != "phoneme":
                text = "".join(text.split());
            else:
                text = text.split()

            ### check illegal words
            illegal_word = False
            for char in text:
                if char not in char2idx:
                    if hp.input_mode == "word" or hp.input_mode == "syllable" or hp.input_mode == "pinyin_syl" or hp.input_mode == "phoneme":
                        # word base or syllable base
                        temp.append(char2idx['oov'])
                    elif hp.input_mode == "bopomofo" or hp.input_mode == "pinyin":
                        # bopomofo base
                        illegal_word = True
                        break
                else:
                    temp.append(char2idx[char])

            if illegal_word:
                continue

            #text = [char2idx[char] for char in text]
            text = temp
            text_lengths.append(len(text))
            if mode=="evaluate":
                texts.append(np.array(text, np.int32))
            else:
                texts.append(np.array(text, np.int32).tostring())

            fpath = os.path.join(hp.data, fname + ".wav")
            fpaths.append(fpath)

        return fpaths, text_lengths, texts
    else:
        lines = open(hp.test_data, 'r').readlines()
        sents = []
        for line in lines:

            if hp.input_mode == "pinyin" or hp.input_mode == "pinyin_syl":
                # pinyin base
                text_pinyin = pinyin(line, style=Style.TONE3)
                line = ""
                for t in text_pinyin:
                    if hp.withtone:
                        if hp.input_mode == "pinyin":
                            line = line + t[0]
                        else:
                            line = line + " " + t[0]
                    else:
                        tmp = ''.join([i for i in t[0] if not i.isdigit()])
                        if hp.input_mode == "pinyin":
                            line = line + tmp
                        else:
                            line = line + " " + t[0]
            elif hp.input_mode == "phoneme":
                t_tmp = ""
                for t in line:
                    if t not in word2phonelist:
                        t_tmp = t_tmp + " " + "oov"
                    else:
                        phonelist = word2phonelist[t]
                        t_tmp = t_tmp + " " + " ".join(phonelist)
                line = t_tmp
            elif hp.input_mode != "word":
                # bopomofo base
                if hp.withtone:
                    line = to_bopomofo(line.strip())
                else:
                    line = to_bopomofo(line.strip(), tones=False)

            if hp.input_mode != "pinyin" and hp.input_mode != "pinyin_syl" and hp.input_mode != "phoneme":
                english_check = re.search('[a-zA-Z]', line)
                if english_check:
                    continue

            if hp.input_mode == "syllable" or hp.input_mode == "pinyin_syl":
                line = line + " E"
                line = line.split()
            elif hp.input_mode == "phoneme":
                line = line + " S"
                line = line.split()
            else:
                line = ''.join(line.split()) + "E"
            sents.append(line)

        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)

        if hp.input_mode == "word" or hp.input_mode == "syllable" or hp.input_mode == "pinyin_syl" or hp.input_mode == "phoneme":
            # word base
            for i, sent in enumerate(sents):
                for j, char in enumerate(sent):
                    if char in char2idx:
                        texts[i, j] = char2idx[char]
                    else:
                        texts[i, j] = char2idx['oov']
        elif hp.input_mode == "bopomofo" or hp.input_mode == "pinyin":
            # bopomofo base
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]

        return texts

    """
    else:
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts
    """

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath.decode())
                mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
                mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels*hp.r))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=4,
                                            capacity=hp.batch_size * 4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

