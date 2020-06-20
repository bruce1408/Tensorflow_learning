#!/usr/bin/env python
# coding: utf-8

# # Sample Hangul RNN

# In[1]:


# -*- coding: utf-8 -*-
# Import Packages
import numpy as np
import tensorflow as tf
import collections
import string
import argparse
import time
import os
from six.moves import cPickle
from TextLoader import *
from Hangulpy import *
print ("Packages Imported")


# # Load dataset using TextLoader

# In[2]:


data_dir    = "data/nine_dreams"
batch_size  = 50
seq_length  = 50
data_loader = TextLoader(data_dir, batch_size, seq_length)
# This makes "vocab.pkl" and "data.npy" in "data/nine_dreams"   
#  from "data/nine_dreams/input.txt" 
vocab_size = data_loader.vocab_size
vocab = data_loader.vocab
chars = data_loader.chars
print ( "type of 'data_loader' is %s, length is %d" 
       % (type(data_loader.vocab), len(data_loader.vocab)) )
print ( "\n" )
print ("data_loader.vocab looks like \n%s " %
       (data_loader.vocab))
print ( "\n" )
print ( "type of 'data_loader.chars' is %s, length is %d" 
       % (type(data_loader.chars), len(data_loader.chars)) )
print ( "\n" )
print ("data_loader.chars looks like \n%s " % (data_loader.chars,))


# # Define Network

# In[3]:


rnn_size   = 512
num_layers = 3
grad_clip  = 5.

_batch_size = 1
_seq_length = 1

vocab_size = data_loader.vocab_size

with tf.device("/cpu:0"):
    # Select RNN Cell
    unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)
    # Set paths to the graph 
    input_data = tf.placeholder(tf.int32, [_batch_size, _seq_length])
    targets    = tf.placeholder(tf.int32, [_batch_size, _seq_length])
    initial_state = cell.zero_state(_batch_size, tf.float32)

    # Set Network
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
            inputs = tf.split(1, _seq_length, tf.nn.embedding_lookup(embedding, input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            
    # Loop function for seq2seq
    def loop(prev, _):
        prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
        prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        return tf.nn.embedding_lookup(embedding, prev_symbol)
    # Output of RNN 
    outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state
                                , cell, loop_function=None, scope='rnnlm')
    output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    # Next word probability 
    probs = tf.nn.softmax(logits)
    # Define LOSS
    loss = tf.nn.seq2seq.sequence_loss_by_example([logits], # Input
        [tf.reshape(targets, [-1])], # Target
        [tf.ones([_batch_size * _seq_length])], # Weight 
        vocab_size)
    # Define Optimizer
    cost = tf.reduce_sum(loss) / _batch_size / _seq_length
    final_state = last_state
    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    _optm = tf.train.AdamOptimizer(lr)
    optm = _optm.apply_gradients(zip(grads, tvars))

print ("Network Ready")


# In[6]:


# Sample ! 
def sample( sess, chars, vocab, __probs, num=200, prime=u'ㅇㅗᴥㄴㅡㄹᴥ '):
    state = sess.run(cell.zero_state(1, tf.float32))
    _probs = __probs
    prime = list(prime)
    for char in prime[:-1]:
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {input_data: x, initial_state:state}
        [state] = sess.run([final_state], feed)

    def weighted_pick(weights):
        weights = weights / np.sum(weights) 
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {input_data: x, initial_state:state}
        [_probsval, state] = sess.run([_probs, final_state], feed)
        p = _probsval[0]
        sample = int(np.random.choice(len(p), p=p))
        # sample = weighted_pick(p)
        # sample = np.argmax(p)
        pred = chars[sample]
        ret += pred
        char = pred
    return ret
print ("sampling function done.")


# # Sample

# In[7]:


save_dir = 'data/nine_dreams'
prime = decompose_text(u"누구 ")

print ("Prime Text : %s => %s" % (automata(prime), "".join(prime)))
n = 2000

sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(tf.all_variables())
ckpt = tf.train.get_checkpoint_state(save_dir)

# load_name = u'data/nine_dreams/model.ckpt-0'
load_name = u'data/nine_dreams/model.ckpt-99000'

print (load_name)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, load_name)
    sampled_text = sample(sess, chars, vocab, probs, n, prime)
    #print ("")
    print (u"SAMPLED TEXT = %s" % sampled_text)
    print ("")
    print ("-- RESULT --")
    print (automata("".join(sampled_text)))


# In[ ]:




