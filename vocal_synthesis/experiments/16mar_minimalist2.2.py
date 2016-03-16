import numpy as np
from scipy.io import wavfile
import os
import sys
#os.chdir("..")
sys.path.append( os.pardir )
#sys.stderr.write("current working directory: %s\n" % os.getcwd())
import cPickle as pickle
from lasagne.updates import *
import rnn_experiment as experiment

if __name__ == "__main__":

    # e.g. 1000_60sec.pkl
    in_pkl = sys.argv[1]
    out_pkl = sys.argv[2]

    with open(in_pkl) as f:
        dat = pickle.load(f)
    X_train, X_valid, X_test = dat[0]

    sys.stderr.write("X_train shape = %s\n" % str(X_train.shape))
    sys.stderr.write("X_valid shape = %s\n" % str(X_valid.shape))
    sys.stderr.write("X_test shape = %s\n" % str(X_test.shape))

    args = dict()
    args["seed"] = 0
    args["batch_size"] = 16
    args["learning_rate"] = 0.01
    args["momentum"] = 0.9
    args["num_epochs"] = 2000
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["X_test"] = X_test
    args["update_method"] = rmsprop
    args["out_pkl"] = out_pkl
    args["in_model"] = "../models/16mar_minimalist2_use_mean.model"
    
    args["config"] = "../configurations/19feb_testing_d_minimalist2.py"

    experiment.train(args)

