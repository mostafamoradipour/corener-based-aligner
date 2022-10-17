import argparse
from os.path import join
import onnx
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from onnx_tf.backend import prepare


parser = argparse.ArgumentParser()
parser.add_argument('--onnx-model', type=str, default='saeid/best.onnx', help='weights path')
opt = parser.parse_args()

onnx_model = onnx.load(opt.onnx_model)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(opt.onnx_model.split('.')[0])
# import_to_tensorboard("model_var.pb", "tb_log")
