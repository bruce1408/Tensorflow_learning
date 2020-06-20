#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Packages
import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle
print ("Packages Imported")


# In[2]:


# Load text
# data_dir    = "data/tinyshakespeare"
data_dir    = "data/linux_kernel"
save_dir    = "data/linux_kernel"
input_file  = os.path.join(data_dir, "input.txt")
with open(input_file, "r") as f:
    data = f.read()
print ("Text loaded from '%s'" % (input_file))


# In[3]:


# Preprocess Text
# First, count the number of characters
counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # <= Sort
print ("Type of 'counter.items()' is %s and length is %d" 
       % (type(counter.items()), len(counter.items()))) 
for i in range(5):
    print ("[%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print (list(counter.items())[i])

print (" ")
print ("Type of 'count_pairs' is %s and length is %d" 
       % (type(count_pairs), len(count_pairs))) 
for i in range(5):
    print ("[%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print (count_pairs[i])


# In[4]:


# Let's make dictionary
chars, counts = zip(*count_pairs)
vocab = dict(zip(chars, range(len(chars))))
print ("Type of 'chars' is %s and length is %d" 
    % (type(chars), len(chars))) 
for i in range(5):
    print ("[%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print ("chars[%d] is '%s'" % (i, chars[i]))
    
print ("")

print ("Type of 'vocab' is %s and length is %d" 
    % (type(vocab), len(vocab))) 
for i in range(5):
    print ("[%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print ("vocab['%s'] is %s" % (chars[i], vocab[chars[i]]))
    
# SAve chars and vocab
with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'wb') as f:
    cPickle.dump((chars, vocab), f)


# # chars[0] converts index to char
# # vocab['a'] converts char to index

# In[5]:


# Now convert all text to index using vocab! 
corpus = np.array(list(map(vocab.get, data)))
print ("Type of 'corpus' is %s, shape is %s, and length is %d" 
    % (type(corpus), corpus.shape, len(corpus)))

check_len = 10
print ("\n'corpus' looks like %s" % (corpus[0:check_len]))
for i in range(check_len):
    _wordidx = corpus[i]
    print ("[%d/%d] chars[%02d] corresponds to '%s'" 
           % (i, check_len, _wordidx, chars[_wordidx]))


# In[6]:


# Generate batch data 
batch_size  = 50
seq_length  = 200
num_batches = int(corpus.size / (batch_size * seq_length))
# First, reduce the length of corpus to fit batch_size
corpus_reduced = corpus[:(num_batches*batch_size*seq_length)]
xdata = corpus_reduced
ydata = np.copy(xdata)
ydata[:-1] = xdata[1:]
ydata[-1]  = xdata[0]
print ('xdata is ... %s and length is %d' % (xdata, xdata.size))
print ('ydata is ... %s and length is %d' % (ydata, xdata.size))
print ("")

# Second, make batch 
xbatches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
ybatches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
print ("Type of 'xbatches' is %s and length is %d" 
    % (type(xbatches), len(xbatches)))
print ("Type of 'ybatches' is %s and length is %d" 
    % (type(ybatches), len(ybatches)))
print ("")

# How can we access to xbatches?? 
nbatch = 5
temp = xbatches[0:nbatch]
print ("Type of 'temp' is %s and length is %d" 
    % (type(temp), len(temp)))
for i in range(nbatch):
    temp2 = temp[i]
    print ("Type of 'temp[%d]' is %s and shape is %s" % (i, type(temp2), temp2.shape,))


# ## Now, we are ready to make our RNN model with seq2seq

# In[7]:


# Important RNN parameters 
vocab_size = len(vocab)
rnn_size   = 128
num_layers = 2
grad_clip  = 5.

# Construct RNN model 
unitcell   = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell       = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets    = tf.placeholder(tf.int32, [batch_size, seq_length])
istate     = cell.zero_state(batch_size, tf.float32)
# Weigths 
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
        inputs = [tf.squeeze(_input, [1]) for _input in inputs]
# Output
def loop(prev, _):
    prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)
"""
    loop_function: If not None, this function will be applied to the i-th output
    in order to generate the i+1-st input, and decoder_inputs will be ignored,
    except for the first element ("GO" symbol).
""" 
outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, istate, cell
                , loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs  = tf.nn.softmax(logits)
# Loss
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], # Input
    [tf.reshape(targets, [-1])], # Target
    [tf.ones([batch_size * seq_length])], # Weight 
    vocab_size)
# Optimizer
cost     = tf.reduce_sum(loss) / batch_size / seq_length
final_state = last_state
lr       = tf.Variable(0.0, trainable=False)
tvars    = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm    = tf.train.AdamOptimizer(lr)
optm     = _optm.apply_gradients(zip(grads, tvars))

print ("Network Ready")


# In[8]:


# Train the model!
num_epochs    = 50
save_every    = 500
learning_rate = 0.002
decay_rate    = 0.97

sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter(save_dir, graph=sess.graph)
saver = tf.train.Saver(tf.all_variables())
init_time = time.time()
for epoch in range(num_epochs):
    # Learning rate scheduling 
    sess.run(tf.assign(lr, learning_rate * (decay_rate ** epoch)))
    state     = sess.run(istate)
    batchidx  = 0
    for iteration in range(num_batches):
        start_time   = time.time()
        randbatchidx = np.random.randint(num_batches)
        xbatch       = xbatches[batchidx]
        ybatch       = ybatches[batchidx]
        batchidx     = batchidx + 1
        
        # Note that, num_batches = len(xbatches)
        # Train! 
        train_loss, state, _ = sess.run([cost, final_state, optm]
            , feed_dict={input_data: xbatch, targets: ybatch, istate: state}) 
        total_iter = epoch*num_batches + iteration
        end_time     = time.time();
        duration     = end_time - start_time
        
        if total_iter % 100 == 0:
            print ("[%d/%d] cost: %.4f / Each batch learning took %.4f sec" 
                   % (total_iter, num_epochs*num_batches, train_loss, duration))
        if total_iter % save_every == 0: 
            ckpt_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step = total_iter)
            # Save network! 
            print("model saved to '%s'" % (ckpt_path)) 


# ### Run the command line
# ##### tensorboard --logdir=/tmp/tf_logs/char_rnn_tutorial
# ### Open http://localhost:6006/ into your web browser

# In[9]:


print ("Done!! It took %.4f second. " %(time.time() - init_time))


# In[ ]:





# In[ ]:




