import cv2
import numpy as np

import tensorflow as tf

import os
import sys

os.environ["PATH"] += os.pathsep + '/home/developer/miniconda3/envs/YOLOv3_TensorFlow/bin/'


def representative_dataset_gen():
    def preprocess(image):
        img = np.array(image, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (img - 127.5) / 128
        x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        return x_in


    for _ in range(10):
        img = cv2.imread("./data/frame306.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.asarray(img, dtype=np.float32)
        img = np.reshape(img,(1,480,640,3))
        yield [img]

model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
# tf.lite.TFLiteConverter
protob_path = './models/' + model_name + '/saved_model/'
tflite_path = './models/' + model_name + '/model.tflite'
converter = tf.lite.TFLiteConverter.from_saved_model(protob_path, input_shapes={"image_tensor": [None, 480, 640, 3]})
# converter = tf.lite.TFLiteConverter.from_saved_model(protob_path, input_shapes={"Placeholder" : [None, 480, 640, 1]})
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
converter.quantized_input_stats = {converter.get_input_arrays()[0] : (0., 1.)}  # mean, std_dev
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)
