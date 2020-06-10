import tensorflow as tf
import numpy as np
# from classify_image import NodeLookup
import os


class NodeLookup(object):
    def __init__(self, label_lookup_path=None):
        self.node_lookup = self.load(label_lookup_path)

    def load(self, label_lookup_path):
        node_id_to_name = {}
        with open(label_lookup_path) as f:
            for index, line in enumerate(f):
                node_id_to_name[index] = line.strip()
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def init_graph(model_name='/home/lf/桌面/data/model.ckpt-100000.pb'):
    with open(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(file_name, node_lookup, sess):
    image_data = open(file_name, 'rb').read()
    softmax_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/predictions/Softmax:0')
    predictions = sess.run(softmax_tensor, {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = node_lookup
    top_k = predictions.argsort()[-12:][::-1]
    top_names = []
    # for node_id in top_k:
    #  human_string = node_lookup.id_to_string(node_id)
    #  top_names.append(human_string)
    #  score = predictions[node_id]
    #  print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    # return predictions, top_k, top_names
    print(file_name, node_lookup.id_to_string(top_k[0]))
    return node_lookup.id_to_string(top_k[0])


file_name = '/home/lf/桌面/Black_Footed_Albatross_0001_796111.jpg'
# test_dir = '/home/gcnan604/devdata/hdwei/TFExamples/plantSeedlings/test'

# label_file, _ = os.path.splitext('my_freeze.pb')
# label_file = label_file + '.label'
label_file = '/home/lf/桌面/data2/label.txt'
node_lookup = NodeLookup(label_file)
sess = tf.Session()
init_graph()
with open('submission.scv', 'a+') as f:
    for test_file in os.listdir(file_name):
        tempfile = os.path.join(file_name, test_file)
        predicetion = run_inference_on_image(tempfile, node_lookup, sess)
        f.write(test_file + ',' + predicetion + '\n')
