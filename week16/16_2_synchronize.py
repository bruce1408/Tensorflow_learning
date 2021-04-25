# 异步分布式训练
# coding=utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 数据的获取不是本章重点，这里直接导入

# 参数定义 (parameter_name, default_value, description)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", "worker", "ps or worker")
tf.app.flags.DEFINE_integer("task_id", 0, "Task ID of the worker/ps running the train")
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps机")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker机，用逗号隔开")

# 全局变量
MODEL_DIR = "./distribute_model_ckpt/"
DATA_DIR = "./data/mnist/"
BATCH_SIZE = 32


# main函数
def main(self):
    # ==========  STEP1: 读取数据  ========== #
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')  # 读取数据

    # ==========  STEP2: 声明集群  ========== #
    # 构建集群ClusterSpec和服务声明
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})  # 构建集群名单
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)  # 声明服务

    # ==========  STEP3: ps机内容  ========== #
    # 分工，对于ps机器不需要执行训练过程，只需要管理变量。server.join()会一直停在这条语句上。
    if FLAGS.job_name == "ps":
        with tf.device("/cpu:0"):
            server.join()

    # ==========  STEP4: worker机内容  ========== #
    # 下面定义worker机需要进行的操作
    is_chief = (FLAGS.task_id == 0)  # 选取task_id=0的worker机作为chief

    # 通过replica_device_setter函数来指定每一个运算的设备。
    # replica_device_setter会自动将所有参数分配到参数服务器上，将计算分配到当前的worker机上。
    device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster=cluster)

    # 这一台worker机器需要做的计算内容
    with tf.device(device_setter):
        # 输入数据
        x = tf.placeholder(name="x-input", shape=[None, 28 * 28], dtype=tf.float32)  # 输入样本像素为28*28
        y_ = tf.placeholder(name="y-input", shape=[None, 10], dtype=tf.float32)  # MNIST是十分类
        # 第一层（隐藏层）
        with tf.variable_scope("layer1"):
            weights = tf.get_variable(name="weights", shape=[28 * 28, 128], initializer=tf.glorot_normal_initializer())
            biases = tf.get_variable(name="biases", shape=[128], initializer=tf.glorot_normal_initializer())
            layer1 = tf.nn.relu(tf.matmul(x, weights) + biases, name="layer1")
        # 第二层（输出层）
        with tf.variable_scope("layer2"):
            weights = tf.get_variable(name="weights", shape=[128, 10], initializer=tf.glorot_normal_initializer())
            biases = tf.get_variable(name="biases", shape=[10], initializer=tf.glorot_normal_initializer())
            y = tf.add(tf.matmul(layer1, weights), biases, name="y")
        pred = tf.argmax(y, axis=1, name="pred")
        global_step = tf.contrib.framework.get_or_create_global_step()  # 必须手动声明global_step否则会报错
        # 损失和优化
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
        loss = tf.reduce_mean(cross_entropy)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
        if is_chief:
            train_op = tf.no_op()

        hooks = [tf.train.StopAtStepHook(last_step=10000)]
        config = tf.ConfigProto(
            allow_soft_placement=True,  # 设置成True，那么当运行设备不满足要求时，会自动分配GPU或者CPU。
            log_device_placement=False,  # 设置为True时，会打印出TensorFlow使用了哪种操作
        )

        # ==========  STEP5: 打开会话  ========== #
        # 对于分布式训练，打开会话时不采用tf.Session()，而采用tf.train.MonitoredTrainingSession()
        # 详情参考：https://www.cnblogs.com/estragon/p/10034511.html
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                checkpoint_dir=MODEL_DIR,
                hooks=hooks,
                save_checkpoint_secs=10,
                config=config) as sess:
            print("session started!")
            start_time = time.time()
            step = 0

            while not sess.should_stop():
                xs, ys = mnist.train.next_batch(BATCH_SIZE)  # batch_size=32
                _, loss_value, global_step_value = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    print("After %d training steps(%d global steps), loss on training batch is %g (%.3f sec/batch)" % (
                    step, global_step_value, loss_value, sec_per_batch))
                step += 1


if __name__ == "__main__":
    tf.app.run()

