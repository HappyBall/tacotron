# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os
import sys
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, load_vocab
from modules import *
from networks import encoder, decoder1, decoder2
from utils import *
from graph import Graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("keep_train", "False", "keep training from existed model or not")

if __name__ == '__main__':
    keep_train = FLAGS.keep_train
    g = Graph(); print("Training Graph loaded")
    logfile = open(hp.logfile, "a")
    saver = tf.train.Saver(max_to_keep=10)
    init = tf.global_variables_initializer()
    # with g.graph.as_default():
    #sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    #with sv.managed_session() as sess:
    with tf.Session() as sess:
        #while 1:
        writer = tf.summary.FileWriter(hp.logdir, graph = sess.graph)

        if keep_train == "True":
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Continue training from existed latest model...")
        else:
            sess.run(init)
            print("Initial new training...")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.num_epochs+1):
            total_loss, total_mel_loss, total_linear_loss = 0.0, 0.0, 0.0
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, gs, l, l_mel, l_linear = sess.run([g.train_op, g.global_step, g.loss, g.loss1, g.loss2])

                total_loss += l
                total_mel_loss += l_mel
                total_linear_loss += l_linear

                if gs % 1000 == 0:
                    # plot the first alignment for logging
                    al = sess.run(g.alignments)
                    plot_alignment(al[0], gs)

            print("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + ", average mel loss: " + str(total_mel_loss/float(g.num_batch)) + ", average linear loss: " + str(total_linear_loss/float(g.num_batch)) + "\n")
            sys.stdout.flush()
            logfile.write("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + ", average mel loss: " + str(total_mel_loss/float(g.num_batch)) + ", average linear loss: " + str(total_linear_loss/float(g.num_batch)) + "\n")

            # Write checkpoint files
            if epoch % 10 == 0:
                #sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))
                saver.save(sess, hp.logdir + '/model_epoch_{}.ckpt'.format(epoch))
                result = sess.run(g.merged)
                writer.add_summary(result, epoch)

        coord.request_stop()
        coord.join(threads)

    print("Done")
