import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
import sys

def get_net(args):
    X_train = args["X_train"]
    num_inputs = args["num_inputs"] # should always be 1
    seq_length = X_train.shape[1] # determined by pkl

    num_hidden_units = 100

    if "seq_length" not in args:
        seq_length = X_train.shape[1]
    else:
        seq_length = args["seq_length"]

    l_input = InputLayer((None, seq_length, num_inputs))
    l_forward = LSTMLayer(l_input, num_units=num_hidden_units)
    l_forward2 = LSTMLayer(l_forward, num_units=num_hidden_units)
    l_forward3 = LSTMLayer(l_forward2, num_units=num_hidden_units)

    """
    In order to connect a recurrent layer to a dense layer, we need to
    flatten the first two dimensions (our "sample dimensions"); this will
    cause each time step of each sequence to be processed independently
    """
    l_shp = ReshapeLayer(l_forward3, (-1, num_hidden_units))
    l_dense = DenseLayer(l_shp, num_units=1, nonlinearity=linear)
    l_out = ReshapeLayer(l_dense, (-1, seq_length, 1))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    return l_out