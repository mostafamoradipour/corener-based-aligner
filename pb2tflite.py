import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--pb-model', type=str, default='saeid/best', help='weights path')
opt = parser.parse_args()

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(opt.pb_model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open(opt.pb_model+'.tflite', 'wb') as f:
  f.write(tflite_model)
