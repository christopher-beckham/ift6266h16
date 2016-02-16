import rnn_experiment as experiment
import numpy as np
from scipy.io import wavfile
import os
import sys
import cPickle as pickle
# ---
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *

def get_basic_net(args):
    X_train = args["X_train"]
    num_inputs = args["num_inputs"] # should always be 1
    seq_length = X_train.shape[1] # determined by pkl
    num_hidden_units = args["num_hidden_units"]
    use_lstm = args["use_lstm"]

    l_input = InputLayer((None, seq_length, num_inputs))
    if use_lstm:
        sys.stderr.write("using lstm layers..\n")
        l_forward = LSTMLayer(l_input, num_units=num_hidden_units)
    else:
        l_forward = RecurrentLayer(l_input, num_units=num_hidden_units)
    """
    In order to connect a recurrent layer to a dense layer, we need to
    flatten the first two dimensions (our "sample dimensions"); this will
    cause each time step of each sequence to be processed independently
    """
    l_shp = ReshapeLayer(l_forward, (-1, num_hidden_units))
    l_dense = DenseLayer(l_shp, num_units=1, nonlinearity=linear)
    l_out = ReshapeLayer(l_dense, (-1, seq_length, 1))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    return l_out

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
args["num_hidden_units"] = 100
args["use_lstm"] = True
args["batch_size"] = 1000
args["learning_rate"] = 0.01
args["momentum"] = 0.9
args["num_epochs"] = 200
args["X_train"] = X_train
args["X_valid"] = X_valid
args["X_test"] = X_test

l_out = get_basic_net(args)
args["l_out"] = l_out

model = experiment.train(args)

print "writing to file: %s" % (out_pkl)
with open(out_pkl, "wb") as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


