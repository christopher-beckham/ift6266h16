import rnn_experiment as experiment
import numpy as np
from scipy.io import wavfile
import os
import sys
import cPickle as pickle
from lasagne.updates import *

if __name__ == "__main__":

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
    args["num_inputs"] = 1
    args["batch_size"] = 64
    args["learning_rate"] = 0.001
    args["momentum"] = 0.9
    args["num_epochs"] = 100
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["X_test"] = X_test
    #args["update_method"] = adam
    
    args["config"] = "basic_net_2000x1.py"

    model = experiment.train(args)

    sys.stderr.write( "writing to file: %s\n" % (out_pkl) )
    with open(out_pkl, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


