import numpy as np
import tensorflow as tf


# 每台机器要做的内容（为了简化，不训练了，只print一下）
c = tf.constant("Hello from server1")

# 集群的名单
cluster = tf.train.ClusterSpec({"local":["localhost:2222", "localhost:2223"]})

# 服务的声明，同时告诉这台机器他是名单中的谁
server = tf.distribute.Server(cluster, job_name="local", task_index=0)

# 以server模式打开会话环境
sess = tf.compat.v1.Session(server.target, config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))
server.join()