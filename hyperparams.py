# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from zhon import zhuyin

class Hyperparams:
    '''Hyper parameters'''

    # pipeline
    withtone = True
    input_mode = "pinyin"
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
    #prepro_path = "/nfs/Athena/yangchiyi/lecture_tts_data/prepro_data"
    prepro_path = "/home/yangchiyi/lecture_tts_data/prepro_data"
    #prepro_path = "/home/yangchiyi/DaAiSermon/prepro_data_nosilence"

    if input_mode == "bopomofo":
        if withtone:
            vocab = zhuyin.characters[:-1] + "E" + "˙ˊˇˋ"
        else:
            vocab = zhuyin.characters[:-1] + "E"
    elif input_mode == "pinyin":
        if withtone:
            vocab = "E abcdefghijklmnopqrstuvwxyz?1234"
        else:
            vocab = "E abcdefghijklmnopqrstuvwxyz?"
    else:
        vocab = zhuyin.characters[:-1] + "E"
        #vocab = zhuyin.characters[:-1] + "E" + "˙ˊˇˋ"    #for previous word base

    # bopomofo base without tone
    #vocab = zhuyin.characters[:-1] + "E"

    # bopomofo base with tone
    #vocab = zhuyin.characters[:-1] + "E" + "˙ˊˇˋ"

    #vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    # data = "/data/private/voice/LJSpeech-1.0"
    # data = "/data/private/voice/nick"
    #data = "/nfs/Athena/yangchiyi/lecture_tts_data/wav_trimmed"
    data = "/home/yangchiyi/lecture_tts_data/wav_trimmed"
    #data = "/home/yangchiyi/DaAiSermon/wav_nosilence"
    test_data = 'test_sentences_chinese.txt'
    max_duration = 10.0
    max_len = 25

    # signal processing
    #sr = 16000 # Sample rate.
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 300 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    #logdir = "/home_local/yangchiyi/tacotron_new_logdir/dataAll_hidden512_epoch500_chinese_withtone"
    logdir = "/home/yangchiyi/tacotron_logdir/chinese_hidden256_epoch500_withouttone_syllable"
    logfile = "/home/yangchiyi/tacotron_new/log/chinese_hidden256_epoch500_withtone_syllable_201805101300.log"
    sampledir = 'samples'
    batch_size = 32
    num_epochs = 500
