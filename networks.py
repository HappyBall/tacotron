# -*- coding: utf-8 -*-
'''
modified from
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)

        #translate_gru_out = gru(prenet_out, num_units=hp.embed_size//2, bidirection=True, scope="enc_gru_1") # (N, T_x, E)
        #translate_gru_out_2 = gru(translate_gru_out, num_units=hp.embed_size, bidirection=True, scope="enc_gru_2") # (N, T_x, 2E)
        #prenet_out_2 = prenet(translate_gru_out_2, is_training=is_training, scope="prenet_2") #(N, T_x, E/2)

        # Encoder CBHG
        ## Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2)
        #enc = conv1d_banks(prenet_out_2, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2)

        ## Max pooling
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2)

        ## Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # (N, T_x, E/2) # residual connections

        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2,
                                 scope='highwaynet_{}'.format(i)) # (N, T_x, E/2)

        ## Bidirectional GRU
        memory = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)

    return memory

def decoder1_scheduled(inputs, memory, is_training=True, scope="decoder1", reuse=None, schedule=1.0):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''

    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training, scope="decoder1_prenet")  # (N, T_y/r, E/2)
        gru_cell_1 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru1")
        gru_cell_2 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru2")
        gru_cell_3 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru3")

        def step(previous_step_output, current_input):
            current_input = current_input[0]
            previous_output = previous_step_output[0][:, -hp.n_mels:]
            previous_output = prenet(previous_output, is_training=is_training, scope="decoder1_prenet", reuse=True)
            previous_context = previous_step_output[1]
            previous_attention_weight = previous_step_output[2]
            previous_state = previous_step_output[3:6]

            if is_training:
                bernoulli_sampler = tf.distributions.Bernoulli(probs=schedule)
                sample = tf.fill(current_input.shape, bernoulli_sampler.sample())
                sample = tf.cast(sample, tf.bool)
                current_input = tf.where(sample, current_input, previous_output)
            else:
                current_input = previous_output

            decoder_input = tf.concat([current_input, previous_context], axis=-1)

            dec, state1 = gru_cell_1(decoder_input, previous_state[0]) # (N, T_y/r, E)
            context_vector, attention_weight = do_attention(dec, memory, previous_attention_weight, hp.embed_size)

            _dec, state2 = gru_cell_2(dec, previous_state[1]) # (N, T_y/r, E)
            dec = _dec + dec
            _dec, state3 = gru_cell_3(dec, previous_state[2]) # (N, T_y/r, E)
            dec = _dec + dec

            mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

            return [mel_hats, context_vector, attention_weight, state1, state2, state3]

        batch_size = tf.shape(inputs)[0]
        init_mel = tf.zeros([batch_size, hp.n_mels*hp.r])

        init_context = tf.zeros([batch_size, memory.get_shape().as_list()[-1]])
        init_attention_weight = tf.zeros(tf.shape(memory)[:2])
        init_attention_weight = tf.concat([tf.ones_like(init_attention_weight[:,:1]), init_attention_weight[:,1:]], axis=1)
        init_state = tf.zeros([batch_size, hp.embed_size])
        init = [init_mel, init_context, init_attention_weight, init_state, init_state, init_state]

        inputs_scan = tf.transpose(inputs, [1,0,2])
        output = tf.scan(step, [inputs_scan], initializer=init)

        mel_hats = tf.transpose(output[0], [1,0,2])
        alignments = tf.transpose(output[2], [1,0,2])

    return mel_hats, alignments

def do_attention(state, memory, prev_weight, attention_hidden_units, memory_length=None, reuse=None):
    """
    bahdanau attention, aka, original attention
    state: [batch_size x hidden_units]
    memory: [batch_size x T x hidden_units]
    prev_weight: [batch_size x T]
    """

    state_proj = tf.layers.dense(state, attention_hidden_units, use_bias=True)
    memory_proj = tf.layers.dense(memory, attention_hidden_units, use_bias=None)
    previous_feat = tf.layers.conv1d(inputs=tf.expand_dims(prev_weight,axis=-1), filters=10, kernel_size=50, padding='same')
    previous_feat = tf.layers.dense(previous_feat, attention_hidden_units, use_bias=None)
    temp = tf.expand_dims(state_proj, axis=1) + memory_proj + previous_feat
    temp = tf.tanh(temp)
    score = tf.squeeze(tf.layers.dense(temp, 1, use_bias=None),axis=-1)

    #mask
    if memory_length is not None:
        mask = tf.sequence_mask(memory_length, tf.shape(memory)[1])
        paddings = tf.cast(tf.fill(tf.shape(score), -2**30),tf.float32)
        score = tf.where(mask, score, paddings)

    weight = tf.nn.softmax(score) #[batch x T]
    context_vector = tf.matmul(tf.expand_dims(weight,1),memory)
    context_vector = tf.squeeze(context_vector,axis=1)

    return context_vector, weight

def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, T_y/r, E)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E)
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)

        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

    return mel_hats, alignments

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, T_y, E*K/2)

        # Max pooling
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, T_y, E*K/2)

        ## Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) # (N, T_y, E/2)

        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2,
                                 scope='highwaynet_{}'.format(i)) # (N, T_y, E/2)

        # Bidirectional GRU
        dec = gru(dec, hp.embed_size//2, bidirection=True) # (N, T_y, E)

        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs
