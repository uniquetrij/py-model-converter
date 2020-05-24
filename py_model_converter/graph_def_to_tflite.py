import tensorflow as tf

import os

from tensorflow_core.lite.python.lite import TFLiteConverter

os.environ["PATH"] += os.pathsep + '/home/developer/miniconda3/envs/YOLOv3_TensorFlow/bin/'

model_name = 'ssd_mobilenet_v2_coco_2018_03_29'

protob_path = './models/'+model_name+'/frozen_inference_graph.pb'
tflite_path = './models/'+model_name+'/frozen_inference_graph.tflite'

input_node_names = ['image_tensor']
output_node_names = ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores']

def representative_dataset_gen():
  for _ in range(100):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = TFLiteConverter.from_frozen_graph(protob_path, input_node_names, output_node_names,
                                                      input_shapes={"image_tensor": [None, 480, 640, 3]})

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen


tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)
