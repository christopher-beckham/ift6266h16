import rnn_experiment as experiment
import numpy as np
from scipy.io import wavfile
import os
import sys
import cPickle as pickle

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
    args["num_inputs"] = 1
    args["batch_size"] = 500
    args["learning_rate"] = 0.01
    args["momentum"] = 0.9
    args["num_epochs"] = 2000
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["X_test"] = X_test
    #args["adagrad"] = True

    args["in_model"] = "models/run_basic_net_d3.model"
    
    #args["config"] = "basic_net.py"
    args["config"] = "basic_net_d3.py"

    model = experiment.train(args)

    sys.stderr.write( "writing to file: %s\n" % (out_pkl) )
    with open(out_pkl, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


