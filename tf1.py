################################################################
# Startup/header code for using TensorFlow v1 for examples in
# Charniak's Introduction to Deep Learning book
################################################################


# Remove startup message - WARNING -- not sure if this is the best log level
# setting.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as layers
# Disable eager execution to permit "Session" objects to run
tf.disable_eager_execution()

# Remove deprecation messages (items below did not work)
#tf.get_logger().setLevel('INFO')
#tf.logging.set_verbosity(tf.logging.ERROR)






