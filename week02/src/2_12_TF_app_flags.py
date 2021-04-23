# coding:utf-8

# 学习使用 tf.app.flags 使用，全局变量
# 可以再命令行中运行也是比较方便，如果只写 python app_flags.py 则代码运行时默认程序里面设置的默认设置
# 若 python app_flags.py --train_data_path <绝对路径 train.txt> --max_sentence_len 100
#    --embedding_size 100 --learning_rate 0.05  代码再执行的时候将会按照上面的参数来运行程序

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_string("train_data_path", "/home/yongcai/chinese_fenci/train.txt", "training data dir")
tf.app.flags.DEFINE_string("log_dirs", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("max_sentence_len", 80, "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 50, "embedding size")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")


def main(unused_argv):
    train_data_path = FLAGS.train_data_path

    print("train_data_path", train_data_path)
    max_sentence_len = FLAGS.max_sentence_len

    print("max_sentence_len", max_sentence_len)
    embdeeing_size = FLAGS.embedding_size

    print("embedding_size", embdeeing_size)
    abc = tf.add(max_sentence_len, embdeeing_size)

    init = tf.compat.v1.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("abc", sess.run(abc))

    # sv = tf.train.Supervisor(logdir=FLAGS.log_dirs, init_op=init)
    # with sv.managed_session() as sess:
    #     print("abc:", sess.run(abc))

        # sv.saver.save(sess, "/home/yongcai/tmp/")


# 使用这种方式保证了，如果此文件被其他文件 import的时候，不会执行main 函数
if __name__ == '__main__':
    tf.compat.v1.app.run()  # 解析命令行参数，调用main 函数 main(sys.argv)
