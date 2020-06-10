import numpy as np
import tensorflow as tf

out = np.array([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
y = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

res = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=out, label_smoothing=0)
print(tf.Session().run(res))

res2 = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=out, label_smoothing=0.001)
print(tf.Session().run(res2))

# new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes

new_onehot_labels = y * (1 - 0.001) + 0.001 / 3
print(y)
print(new_onehot_labels)
res3 = tf.losses.softmax_cross_entropy(onehot_labels=new_onehot_labels, logits=out, label_smoothing=0)
print(tf.Session().run(res3))