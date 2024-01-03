################################################################
# bonus.py
#
# A4 bonus question -- using Fashion MNIST data
#
################################################################

from a4_program import *

if __name__ == "__main__":
    # First time: downloads data for fashion MNIST, stores in a local directory on your machine
    # Second and later times: will load the local copy of the dataset on your machine
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Add code below and any additional functions above to run your a4_program models on fashion MNIST
    # Information about fashion MNIST may be found here: https://www.tensorflow.org/tutorials/keras/classification
    # (Use/adapt code and low-level TF operations for this; do not submit keras code for the assignment)



