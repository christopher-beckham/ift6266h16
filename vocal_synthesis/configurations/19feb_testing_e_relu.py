import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
import sys

def get_net(args):
    # e.g. (bs, 50, 1800)
    X_train = args["X_train"]
    num_inputs = X_train.shape[2]
    seq_length = X_train.shape[1] # determined by pkl

    num_hidden_units = 150

    if "seq_length" not in args:
        seq_length = X_train.shape[1]
    else:
        seq_length = args["seq_length"]

    grad_clip=10

    l_input = InputLayer((None, seq_length, num_inputs))
    l_forward = LSTMLayer(
	l_input, num_units=num_hidden_units, unroll_scan=False, precompute_input=True, nonlinearity=rectify, grad_clipping=grad_clip, gradient_steps=10
    )

    """
    In order to connect a recurrent layer to a dense layer, we need to
    flatten the first two dimensions (our "sample dimensions"); this will
    cause each time step of each sequence to be processed independently
    """
    l_shp = ReshapeLayer(l_forward, (-1, num_hidden_units))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=rectify)
    l_out = ReshapeLayer(l_dense, (-1, seq_length, num_inputs))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    return l_out
