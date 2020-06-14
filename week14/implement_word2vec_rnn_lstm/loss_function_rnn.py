# coding=utf-8
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(1)
import numpy as np
import tensorflow as tf
tf.set_random_seed(0)
np.random.seed(0)


def sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True, softmax_loss_function=None,
                             name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (labels, logits) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
        **Note that to avoid confusion, it is required for the function to accept
        named arguments.**
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    # 此三者都是列表，长度都应该相同
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with tf.name_scope(name, "sequence_loss_by_example",
                       logits + targets + weights):  # 命名空间删掉也可以
        log_perp_list = []
        # 计算每个时间片的损失
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # 默认使用sparse
                target = tf.reshape(target, [-1])
                # target = tf.Print(target, [target], name='target', summarize=4)
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit)
                crossent = tf.Print(crossent, [crossent], name='corss', summarize=4)  # 1x4的矩阵
            else:
                crossent = softmax_loss_function(labels=target, logits=logit)
            log_perp_list.append(crossent * weight)
        # 把各个时间片的损失加起来
        log_perps = tf.add_n(log_perp_list)
        # log_perps = tf.Print(log_perps, [log_perps], name='log', summarize=4)
        # 对各个时间片的损失求平均数
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
        return log_perps


"""
考虑many2many形式的RNN用法，每次输入一个就会得到一个输出
这些输出需要计算平均损失，我们可以指定：
* 每个样本的权重
* 每个时间片的权重
"""
sample_count = 4
target_count = 3
frame_count = 2
# 各个时间片我的答案
logits = [tf.random_uniform((sample_count, target_count)) for i in range(frame_count)]
# 各个时间片的真正答案
targets = [tf.constant(np.random.randint(0, target_count, (sample_count,))) for i in range(frame_count)]
# 每个时间片，每个样本的权重。利用weights我们可以指定时间片权重和样本权重
weights = [tf.ones((sample_count), dtype=tf.float32) * (i + 1) for i in range(frame_count)]
loss1 = sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True)
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, targets, weights, True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y, = sess.run([loss, loss1])
    print(x)
    print(y)
    print(x.shape, y.shape)
    print(sess.run(targets))



