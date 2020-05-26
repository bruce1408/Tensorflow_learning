import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集
mnist = input_data.read_data_sets("../.../MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 10
model_path = "./8_6/model.ckpt"

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2st layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="input_x")
y = tf.placeholder(tf.float32, [None, n_classes], name="input_y")

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x, weights, biases):
    # layer1
    h1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    h1 = tf.nn.relu(h1)

    # layer2
    h2 = tf.add(tf.matmul(h1, weights['h2']), biases['b2'])
    h2 = tf.nn.relu(h2)

    # out
    out = tf.add(tf.matmul(h2, weights['out']), biases['out'])

    return out


# Construct model
logits = multilayer_perceptron(x, weights, biases)
pred = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

corrcet_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(corrcet_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# 保存模型
saver = tf.train.Saver()
tf.add_to_collection("pred", pred)
tf.add_to_collection('acc', accuracy)

with tf.Session() as sess:
    sess.run(init)

    step = 0
    while step * batch_size < 1000:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={x: batch_xs, y: batch_ys})
        if step % display_step == 0:
            # step: 1790 loss: 16.9724 acc: 0.95
            print("step: ", step, "loss: ", loss, "acc: ", acc)
            saver.save(sess, save_path=model_path, global_step=step)
        step += 1
    print("Train Finish!")
