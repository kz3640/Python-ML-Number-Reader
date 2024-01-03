################################################################
# a4_program.py
#
# Programs from Eugene Charniak's Intro to DL, Chs 2-3
#
################################################################


from debug import *
pTimer = DebugTimer("A4 Timings")

from tf1 import tf, layers
from input_data import mnist
from mnist_vis import *
from a4_models import *

import sys
from tqdm import tqdm
import argparse
import random
import numpy as np
import statistics

# Global Vars
TEST_SIZE = 10000
RANDOM_SEED = 7 # for luck.

pTimer.qcheck("Initialization")

################################################################
# Utilities
################################################################

def set_seed( use_seed ):
    """ Set a random seed if use_seed is True (default). """
    # DEBUG: Set seed for (1) tensorflow, (2) random and (3) numpy for stable results
    if use_seed:
        tf.random.set_random_seed( RANDOM_SEED )
        random.seed( RANDOM_SEED )
        np.random.seed( RANDOM_SEED )


################################################################
# Train and Compute Test Error on a Network
################################################################
def train_run_nn( timer, model, rounds, numBatches, learningRate ):
    """ 
    Figure 2.2 : network construction function is passed as the 'model' parameter.
                 default is a single linear feature layer followed by softmax, as
                 given in Figure 2.2.

                 Some modifications have been made to the original code.
    """

    # Define inputs for images and 1-hot encoded target vectors ('answers')
    # Construct neural network model using inputs
    img = tf.placeholder(tf.float32, [100, 784])
    ans = tf.placeholder(tf.float32, [100, 10])
    ( prbs, model_name ) =  model(img)

    # Define cross-entropy loss node 'xEnt' and gradient descent node 'train'
    xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(learningRate).minimize(xEnt)

    # Compute number of correct samples (Modified from book)
    correctOutput = tf.equal(  tf.argmax(prbs,1), tf.argmax(ans,1))
    correctSamples = tf.reduce_sum(tf.cast(correctOutput, tf.float32))

    # Start session, initialize tf graph variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #-----------------------------------------------------
    accList = []
    print(">>> Training model (" + str(rounds) + " rounds): " + model_name )
    for r in tqdm( range(0,rounds) ):
        for i in range(numBatches):
            imgs, anss = mnist.train.next_batch(100)
            sess.run(train, feed_dict={img: imgs, ans: anss})

        # Run over MNIST test data only once, and compute accuracy more simply
        # WARNING: Assumes batch size evenly divides the test set!
        sumCorrect=0
        for i in range(TEST_SIZE // 100):
            imgs, anss = mnist.test.next_batch(100)
            sumCorrect+=sess.run(correctSamples,feed_dict={img: imgs, ans: anss})

        # Output model parameters, accuracy, timing information
        accuracy = sumCorrect / TEST_SIZE * 100
        accList.append(accuracy)
    
    timer.qcheck(model_name + ": " + str(rounds) + " train/test rounds") 
    
    # Report statistics on test accuracy for multiple rounds
    print("\n>>> Testing Model: " + model_name + \
            "\n    Rounds: " + str(rounds) + ", Batch Size: 100," + \
            " Batches: " + str(numBatches) + " Learning rate: " + str(learningRate) + "\n" + \
              "    Test Accuracies: " + str(accList) + "\n" + \
              "    (MAX, MIN): (%.2f, " % max(accList) + "%.2f)" % min(accList) + "\n" + \
                "    AVERAGE: %.2f" % (sum(accList) / len(accList) ) + "%")
    if rounds > 1:
        print("    STDEV: %.2f" % statistics.stdev(accList) + "%\n")
    else:
        print("")

    
    sess.close()


################################################################
# Main program
################################################################

if __name__ == "__main__":
    # Command line arguments and default parameters
    parser = argparse.ArgumentParser(
            description="Program for Assignment 4, CSCI 335 (Machine Learning), Fall 2022. Based on Intro to Deep Learning (Charniak) Chs 2-3", 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models", default="[ Ls, lr_Ls, LrLs, L2s ]", help="{all, linear, conv, lr_all, lr_linear, lr_conv}\nor list (e.g., '[ Ls, lr_Ls ]' )")
    parser.add_argument("--lrate", type=float, default=0.5, help="learning rate")
    parser.add_argument("--batches", type=int, default=1000, help="number of batches")
    parser.add_argument("--rounds", type=int, default=1, help="number of train/test rounds")
    parser.add_argument("--no-rseed", dest="no_rseed", action="store_true", 
                        help="do not use random seed")
    parser.set_defaults(no_rseed=False)
    
    # Process command line args, set random seeds unless told not to
    args = parser.parse_args()

    # Model sets
    linear_models = [ Ls, LrLs, L2s ]
    conv_models = [ CrLs, CrCLs ]

    linear_layer = [ lr_Ls, lr_LrLs, lr_L2s ]
    conv_layer = [ lr_CrLs, lr_CrCLs ] 

    all_models = linear_models + conv_models 
    all_layer = linear_layer + conv_layer 

    model_dict = { "all": all_models, "linear": linear_models, "conv": conv_models,
                   "lr_all": all_layer,  "lr_linear": linear_layer, "lr_conv": conv_layer }

    # Select models to run
    if args.models in model_dict:
        model_list = model_dict[ args.models ]
    else:
        # Read list of model function names
        model_list = eval( args.models )

    # Run the models
    for nn_model in model_list:
        tf.reset_default_graph() # DEBUG: Clear tensorflow graph

        # DEBUG: Need to reset random seeds at this level.
        if not args.no_rseed:
            set_seed( not args.no_rseed)

        train_run_nn(pTimer, 
                     model=nn_model, 
                     rounds=args.rounds,
                     numBatches=args.batches, 
                     learningRate=args.lrate )

    print( pTimer )

    #view_digit( 0, mnist.train.images, mnist.train.labels )
    #view_array( mnist.train.images[:4,:], mnist.train.labels[:4], 2 )

