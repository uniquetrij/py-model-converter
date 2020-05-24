# # import numpy as np
# # import tensorflow as tf
# #
# # # Load TFLite model and allocate tensors.
# # interpreter = tf.lite.Interpreter(model_path="./models/crowd-count.tflite")
# # interpreter.allocate_tensors()
# #
# # # Get input and output tensors.
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()
# #
# # # Test model on random input data.
# # input_shape = input_details[0]['shape']
# # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# # interpreter.set_tensor(input_details[0]['index'], input_data)
# #
# # interpreter.invoke()
# #
# # # The function `get_tensor()` returns a copy of the tensor data.
# # # Use `tensor()` in order to get a pointer to the tensor.
# # output_data = interpreter.get_tensor(output_details[0]['index'])
# # print(output_data)
import numpy as np
#
import cv2
from PIL import ImageDraw, Image, ImageFont
from tensorflow.lite.python.interpreter import Interpreter
feed_vid = cv2.VideoCapture(-1)

interpreter = Interpreter(model_path="./models/crowd_count/model1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess(image):
    img = np.array(image, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (img - 127.5) / 128
    x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    return x_in


font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 25, encoding="unic")
while True:
    success, im = feed_vid.read()
    if not success:
        continue

    image = preprocess(im)
    input_data = image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    sum = np.absolute(np.int32(np.sum(output_data)))

    im_pil = Image.fromarray(im)
    draw = ImageDraw.Draw(im_pil, "RGBA")
    draw.rectangle(((0, 0), (im.shape[1], 30)), fill=(255, 0, 0, 128))
    draw.text((10, 0), 'Queue: ' + str(sum), (255, 255, 255), font=font)
    im = np.array(im_pil)
    cv2.imshow("image", im)
    cv2.waitKey(1)


# import time
#
# import numpy as np
# from PIL import Image
# from tensorflow.lite.python.interpreter import Interpreter
#
#
# def preprocess( img):
#     img = (img - 127.5) / 128
#     x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
#     return x_in
#
# im = Image.open("./data/frame306.jpg").convert('L')
# image = np.array(im, dtype=np.float32)/255
# image = preprocess(image)
#
# # Load TFLite model and allocate tensors.
#
# interpreter = Interpreter(model_path="./models/crowd-count.tflite")
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Test model on random input data.
# input_shape = input_details[0]['shape']
#
# # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# input_data =image
# start = time.time()
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# sum = np.absolute(np.int32(np.sum(output_data)))
# print("time {}",time.time() - start)
# print(sum)
