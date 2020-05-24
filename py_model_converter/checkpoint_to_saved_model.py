import tensorflow as tf
from tensorflow.python.saved_model import builder, tag_constants, signature_constants, signature_def_utils

# model_name = "faster_rcnn_inception_v2_coco_2018_01_28"
#
# meta_path = './models/' + model_name + '/model.ckpt.meta'
# ckpt_path = './models/' + model_name + '/model.ckpt'
# save_path = './models/' + model_name + '/saved_model'
#
# input_node_names = ['image_tensor']
# output_node_names = ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores']

model_name = "yolov3"

meta_path = './models/' + model_name + '/yolov3.ckpt.meta'
ckpt_path = './models/' + model_name + '/yolov3.ckpt'
save_path = './models/' + model_name + '/saved_model'
#
# input_node_names = ['Placeholder']
# output_node_names = ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores']

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, ckpt_path)

    for x in tf.get_default_graph().get_operations():
            print(x)


    # ins = {key: value for (key, value) in
    #        [(x, tf.get_default_graph().get_tensor_by_name(x + ':0')) for x in input_node_names]}
    # outs = {key: value for (key, value) in
    #         [(x, tf.get_default_graph().get_tensor_by_name(x + ':0')) for x in output_node_names]}
    # #
    # # tf.saved_model.simple_save(sess, save_path, ins, outs)
    #
    # signature_def_map = {
    #     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #         signature_def_utils.predict_signature_def(ins, outs)
    # }
    #
    # b = builder.SavedModelBuilder(save_path)
    # b.add_meta_graph_and_variables(
    #     sess,signature_def_map=signature_def_map,
    #     tags=[tag_constants.SERVING])
    # b.save()
