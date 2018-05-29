# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tensorflow as tf
from modules import *
from networks import encoder, decoder1, decoder2
from utils import *
from data_load import load_vocab, get_batch

class Graph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        is_training=True if mode=="train" else False

        # Graph
        # Data Feeding
        # x: Text. (N, Tx)
        # y: Reduced melspectrogram. (N, Ty//r, n_mels*r)
        # z: Magnitude. (N, Ty, n_fft//2+1)
        if mode=="train":
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
        elif mode=="eval":
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))
        else: # Synthesize
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))

        # Get encoder/decoder inputs
        self.encoder_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, T_x, E)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)

        # Networks
        with tf.variable_scope("net"):
            # Encoder
            self.memory = encoder(self.encoder_inputs, is_training=is_training) # (N, T_x, E)

            # Decoder1
            self.y_hat, self.alignments = decoder1(self.decoder_inputs,
                                                     self.memory,
                                                     is_training=is_training) # (N, T_y//r, n_mels*r)

            # Guided attention loss
            batch_size, N, T = tf.shape(self.alignments)[0], tf.shape(self.alignments)[1], tf.shape(self.alignments)[2]
            g = 0.2
            Ns = tf.tile(tf.expand_dims(tf.range(N)/N, 1), [1, T]) # shape: [N, T]
            Ts = tf.tile(tf.expand_dims(tf.range(T)/T, 0), [N, 1]) # shape: [N, T]
            W = tf.ones([N, T]) - tf.exp(-1*(tf.cast(tf.square(Ns - Ts), tf.float32) / (2*tf.square(g))))
            nearly_diagonal_constraint = tf.multiply(self.alignments, tf.tile(tf.expand_dims(W, 0), [batch_size, 1, 1]))
            self.guided_attn_loss = tf.reduce_mean(nearly_diagonal_constraint)

            # Decoder2 or postprocessing
            self.z_hat = decoder2(self.y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)

        # monitor
        self.audio_h = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)
        if mode == "train":
            self.audio_gt = tf.py_func(spectrogram2wav, [self.z[0]], tf.float32)

        if mode in ("train", "eval"):
            # Loss
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            #self.loss = self.loss1 + self.loss2
            self.loss = self.loss1 + self.loss2 + self.guided_attn_loss

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            tf.summary.scalar('{}/guided_attention_loss'.format(mode), self.guided_attn_loss)
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss2'.format(mode), self.loss2)
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)
            tf.summary.image("{}/attention".format(mode), tf.expand_dims(self.alignments, -1), max_outputs=1)

            tf.summary.audio("{}/sample_hat".format(mode), tf.expand_dims(self.audio_h, 0), hp.sr)
            tf.summary.audio("{}/sample_gt".format(mode), tf.expand_dims(self.audio_gt, 0), hp.sr)
            self.merged = tf.summary.merge_all()
