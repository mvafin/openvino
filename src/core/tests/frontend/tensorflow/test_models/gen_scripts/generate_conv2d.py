import numpy as np
import os
import sys
import tensorflow as tf

tf.compat.v1.reset_default_graph()

# Create the graph and model
with tf.compat.v1.Session() as sess:
    input = tf.compat.v1.placeholder(tf.float32, [1, 5, 5, 1], 'x')

    kernel = tf.constant(np.random.randn(2, 2, 1, 2), dtype=tf.float32)

    tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID')

    tf.compat.v1.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "conv2d"), 'conv2d.pb', False)
