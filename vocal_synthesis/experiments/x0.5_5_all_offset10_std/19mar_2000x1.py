
import sys
sys.path.append("../..")
import rnn_experiment_2 as experiment
import numpy as np
from scipy.io import wavfile
import os
#import sys
import cPickle as pickle
from lasagne.updates import *

if __name__ == "__main__":

    # e.g. 1000_60sec.pkl
    in_pkl = sys.argv[1]
    out_pkl = sys.argv[2]

    if ".pkl" in in_pkl:
    	with open(in_pkl) as f:
            dat = pickle.load(f)
    	X_train, X_valid, X_test = dat[0]
    else:
        ctr = np.load(in_pkl)
        X_train, X_valid, X_test = ctr["arr_0"], ctr["arr_1"], ctr["arr_2"]

    sys.stderr.write("X_train shape = %s\n" % str(X_train.shape))
    sys.stderr.write("X_valid shape = %s\n" % str(X_valid.shape))
    sys.stderr.write("X_test shape = %s\n" % str(X_test.shape))

    args = dict()
    args["seed"] = 0
    args["batch_size"] = 128
    args["learning_rate"] = 0.01
    args["momentum"] = 0.9
    args["num_epochs"] = 4000
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["X_test"] = X_test
    args["update_method"] = rmsprop
    args["out_pkl"] = out_pkl
    args["units"] = [2000]
    #args["out_nonlinearity"] = "sigmoid"
    args["forget_gate"] = 1.0
    
    args["config"] = "../../configurations/19mar_variable_3.py"

    experiment.train(args)

    #sys.stderr.write( "writing to file: %s\n" % (out_pkl) )
    #with open(out_pkl, "wb") as f:
    #    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


