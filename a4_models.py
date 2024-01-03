################################################################
# a4_models.py
#
# Neural Network Models 
#
# Named by sequence of layers:
# - L: linear
# - C: convolutional
# - r: ReLU activations
# - s: softmax normalization
# - L2: indicates 2 linear layers with no activation between
# - lr_: indicates layers package used for implementation
################################################################

from tf1 import tf, layers

# Global definitions
RAND_INIT=tf.random_normal_initializer(stddev=.1)

################################################################
# Linear models
################################################################

def Ls(img):
    """ Fig 2.2: Single linear layer + softmax """
    W = tf.Variable(tf.random_normal([784, 10],stddev=.1))
    b = tf.Variable(tf.random_normal([10], stddev=.1))
    prbs =  tf.nn.softmax( tf.matmul(img, W) + b )

    return (prbs, "Single linear")

def LrLs(img):
    """ Fig 2.9: Double linear layer, using ReLU then softmax. """

    prbs = None

    U = tf.Variable(tf.random_normal([784,784], stddev=.1))
    bU = tf.Variable(tf.random_normal([100,784], stddev=.1))
    V = tf.Variable(tf.random_normal([784,10], stddev=.1))
    bV = tf.Variable(tf.random_normal([10], stddev=.1))
    L10Output = tf.matmul(img,U)+bU
    L10Output = tf.nn.relu(L10Output)
    prbs=tf.nn.softmax(tf.matmul(L10Output,V)+bV)

    return (prbs, "Double linear")
def L2s(img):
    """ Fig 2.9: Double linear layer (NO Relu) then softmax. """

    prbs = None

    U = tf.Variable(tf.random_normal([784, 784], stddev=.1))
    bU = tf.Variable(tf.random_normal([100, 784], stddev=.1))
    V = tf.Variable(tf.random_normal([784, 10], stddev=.1))
    bV = tf.Variable(tf.random_normal([10], stddev=.1))

    L10Output = tf.matmul(img, U) + bU
    prbs = tf.nn.softmax(tf.matmul(L10Output, V) + bV)
    return (prbs, "Double linear -- no ReLU")
################################################################
# Convolutional models
################################################################
def CrLs(img):
    """ Fig 3.7: Single convolution layer + linear layer, using ReLU then softmax. """

    image = tf.reshape(img, [100,28,28,1])
        #Turns img into 4d Tensor
    flts=tf.Variable(tf.truncated_normal([4,4,1,4],stddev=0.1))
        #Create parameters for the filters
    convOut = tf.nn.conv2d(image, flts, [1, 2, 2, 1], "SAME")
        #Create graph to do convolution
    convOut= tf.nn.relu(convOut)
        #Don't forget to add nonlinearity
    convOut=tf.reshape(convOut,[100,784])
        #Back to 100 1d image vectors
    b = tf.Variable(tf.random_normal([10], stddev=.1))
    W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
    prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)

    return (prbs, "Single convolution + linear")
def CrCLs(img):
    """ 
    Fig 3.8: Two convolution layers + linear layer. ReLU between convolution layers, linear layer
    to compute class scores after second convolution layer (logits) before using softmax.
    """


    image = tf.reshape(img, [100,28,28,1])
        #Turns img into 4d Tensor
    flts=tf.Variable(tf.truncated_normal([4,4,1,16],stddev=0.1))
        #Create parameters for the filters
    convOut = tf.nn.conv2d(image, flts, [1, 2, 2, 1], "SAME")
        #Create graph to do convolution
    convOut= tf.nn.relu(convOut)
        #Don't forget to add nonlinearity
    poolOut = tf.nn.max_pool(convOut, [1,2,2,1], [1,2,2,1], "SAME")
    flts2=tf.Variable(tf.random.normal([2, 2, 16, 32], stddev=0.1))
    convOut2 = tf.nn.conv2d(poolOut, flts2, [1, 2, 2, 1], "SAME")
    convOut2 = tf.reshape(convOut2,[100,1568])
        #Back to 100 1d image vectors
    b = tf.Variable(tf.random_normal([10], stddev=.1))
    W = tf.Variable(tf.random_normal([1568, 10], stddev=.1))
    prbs = tf.nn.softmax(tf.matmul(convOut2, W) + b)

    return (prbs, "Double convolution + linear")
################################################################
# Linear and Convolutional Models using 'dense' and 'conv2d'
# Layers
################################################################
def lr_Ls(img):
    """ (layers) Fig 2.2: Single linear layer + softmax """
    L1Output = layers.dense(img,
                        10,
                        kernel_initializer=RAND_INIT,
                        bias_initializer=RAND_INIT)
    prbs = tf.nn.softmax( L1Output )
    return (prbs, "(layers) Single linear")

def lr_LrLs(img):
    """ (layers) Fig 2.9: Double linear layer, using ReLU then softmax. """
    image = tf.reshape(img, [100, 28, 28, 1])
    #flts = tf.compat.v1.get_variable("filters", shape=[4, 4, 1, 4],
                                     #initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    convOut = tf.compat.v1.layers.conv2d(inputs=image, filters=4, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
    convOut = tf.reshape(convOut, [100, 784])
    b = tf.Variable(tf.random_normal([10], stddev=0.1))
    W = tf.Variable(tf.random_normal([784, 10], stddev=0.1))
    prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)
    return (prbs, "(layers) Double linear")

def lr_L2s(img):
    """ (layers) Fig 2.9: Double linear layer (NO Relu) then softmax. """
    image = tf.reshape(img, [100, 28, 28, 1])
    #flts = tf.compat.v1.get_variable("filters", shape=[4, 4, 1, 4],
                                     #initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    convOut = tf.compat.v1.layers.conv2d(inputs=image, filters=4, kernel_size=4, strides=2, padding='SAME')
    convOut = tf.reshape(convOut, [100, 784])
    b = tf.Variable(tf.random_normal([10], stddev=0.1))
    W = tf.Variable(tf.random_normal([784, 10], stddev=0.1))
    prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)

    return (prbs, "(layers) Double linear -- no ReLU")


def lr_CrLs(img):
    """ (layers) Fig 3.7: Single convolution layer + linear layer, using ReLU then softmax. """
    image = tf.reshape(img, [100, 28, 28, 1])
    conv = tf.compat.v1.layers.conv2d(inputs=image, filters=4, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
    convOut = tf.reshape(conv, [100, 784])
    b = tf.Variable(tf.random_normal([10], stddev=.1))
    W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
    prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)
    return (prbs, "(layers) Single convolution + linear")


def lr_CrCLs(img):
    """ 
    (layers) Fig 3.8: Two convolution layers + linear layer. ReLU between convolution layers, linear layer
    to compute class scores after second convolution layer (logits) before using softmax.
    """
    image = tf.reshape(img, [100, 28, 28, 1])
    conv1 = tf.compat.v1.layers.conv2d(inputs=image, filters=16, kernel_size=4, strides=2, padding='SAME',
                                       activation=tf.nn.relu)
    conv2 = tf.compat.v1.layers.conv2d(inputs=conv1, filters=32, kernel_size=2, strides=2, padding='SAME',
                                       activation=tf.nn.relu)
    convOut = tf.reshape(conv2, [100, 1568])
    b = tf.Variable(tf.random.normal([10], stddev=0.1))
    W = tf.Variable(tf.random.normal([1568, 10], stddev=0.1))
    prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)

    return (prbs, "(layers) Double convolution + linear")





