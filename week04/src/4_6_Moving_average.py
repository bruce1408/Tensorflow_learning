import tensorflow as tf

# 定义一个变量用于计算滑动平均,初始值为0。
v1 = tf.Variable(0, dtype=tf.float32)

# step变量模拟神经网络中迭代的轮数, 可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类(class),初始化时给定了衰减率(0.99)和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量的滑动平均操作。这里需要给定一个列表,每次执行这个操作时,这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()

    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后的变量取值。在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1, ema.average(v1)]))

    # 更新v1的值为5
    sess.run(tf.assign(v1, 5))

    # 更新v1的滑动平均值。衰减率为min{0.99, (1+step)/(10+step) = 0.1} = 0.1
    sess.run(maintain_averages_op)

    print(sess.run([v1, ema.average(v1)]))
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

