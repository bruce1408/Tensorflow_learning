#!/usr/bin/env python
# coding: utf-8

# # TRAIN HANGUL-RNN

# In[1]:


# -*- coding: utf-8 -*-
# Import Packages
import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle
from TextLoader import *
from Hangulpy import *
print ("Packages Imported")


# # LOAD DATASET WITH TEXTLOADER

# In[2]:


data_dir    = "data/nine_dreams"
batch_size  = 50
seq_length  = 50
data_loader = TextLoader(data_dir, batch_size, seq_length)
# This makes "vocab.pkl" and "data.npy" in "data/nine_dreams"   
#  from "data/nine_dreams/input.txt" 


# # VOCAB AND CHARS

# In[3]:


vocab_size = data_loader.vocab_size
vocab = data_loader.vocab
chars = data_loader.chars
print ( "type of 'data_loader.vocab' is %s, length is %d" 
       % (type(data_loader.vocab), len(data_loader.vocab)) )
print ( "type of 'data_loader.chars' is %s, length is %d" 
       % (type(data_loader.chars), len(data_loader.chars)) )


# # VOCAB: DICTIONARY (CHAR->INDEX)

# In[4]:


print (data_loader.vocab)


# # CHARS: LIST (INDEX->CHAR)

# In[5]:


print (data_loader.chars)
# USAGE
print (data_loader.chars[0])


# # TRAINING BATCH (IMPORTANT!!)

# In[7]:


x, y = data_loader.next_batch()
print ("Type of 'x' is %s. Shape is %s" % (type(x), x.shape,))
print ("x looks like \n%s" % (x))
print
print ("Type of 'y' is %s. Shape is %s" % (type(y), y.shape,))
print ("y looks like \n%s" % (y))


# # DEFINE A MULTILAYER LSTM NETWORK

# In[8]:


rnn_size   = 512
num_layers = 3
grad_clip  = 5. # <= GRADIENT CLIPPING (PRACTICALLY IMPORTANT)
vocab_size = data_loader.vocab_size

# SELECT RNN CELL (MULTI LAYER LSTM)
unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)

# Set paths to the graph
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets    = tf.placeholder(tf.int32, [batch_size, seq_length])
initial_state = cell.zero_state(batch_size, tf.float32)

# Set Network
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(
                embedding, input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
print ("Network ready")


# # Define functions

# In[9]:


# Output of RNN
outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state
                        , cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

# Next word probability
probs = tf.nn.softmax(logits)
print ("FUNCTIONS READY")


# # DEFINE LOSS FUNCTION 

# In[10]:


loss = tf.nn.seq2seq.sequence_loss_by_example([logits], # Input
    [tf.reshape(targets, [-1])], # Target
    [tf.ones([batch_size * seq_length])], # Weight
    vocab_size)
print ("LOSS FUNCTION")


# # DEFINE COST FUNCTION 

# In[11]:


cost = tf.reduce_sum(loss) / batch_size / seq_length

# GRADIENT CLIPPING ! 
lr = tf.Variable(0.0, trainable=False) # <= LEARNING RATE 
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm = tf.train.AdamOptimizer(lr)
optm = _optm.apply_gradients(zip(grads, tvars))

final_state = last_state
print ("NETWORK READY")


# # OPTIMIZE NETWORK WITH LR SCHEDULING

# In[ ]:


num_epochs    = 500
save_every    = 1000
learning_rate = 0.0002
decay_rate    = 0.97

save_dir = 'data/nine_dreams'
sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter(save_dir
                    , graph=sess.graph)
saver = tf.train.Saver(tf.all_variables())
for e in range(num_epochs): # for all epochs

    # LEARNING RATE SCHEDULING 
    sess.run(tf.assign(lr, learning_rate * (decay_rate ** e)))

    data_loader.reset_batch_pointer()
    state = sess.run(initial_state)
    for b in range(data_loader.num_batches):
        start = time.time()
        x, y = data_loader.next_batch()
        feed = {input_data: x, targets: y, initial_state: state}
        # Train!
        train_loss, state, _ = sess.run([cost, final_state, optm], feed)
        end = time.time()
        # PRINT 
        if b % 100 == 0:
            print ("%d/%d (epoch: %d), loss: %.3f, time/batch: %.3f"  
                   % (e * data_loader.num_batches + b
                    , num_epochs * data_loader.num_batches
                    , e, train_loss, end - start))
        # SAVE MODEL
        if (e * data_loader.num_batches + b) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path
                       , global_step = e * data_loader.num_batches + b)
            print("model saved to {}".format(checkpoint_path))


# In[ ]:


# IT TAKE A LOOOOOOOOT OF TIME

