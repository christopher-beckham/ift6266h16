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

    units = args["units"]

    if "seq_length" not in args:
        seq_length = X_train.shape[1]
    else:
        seq_length = args["seq_length"]

    if "nonlinearity" not in args:
        nonlinearity = tanh
    else:
        if args["nonlinearity"] == "relu":
            nonlinearity = rectify
            sys.stderr.write("using relu nonlinearity\n")

    l_input = InputLayer((None, seq_length, num_inputs))
    for unit in units:
    	l_forward = LSTMLayer(l_input, num_units=unit, unroll_scan=False, precompute_input=True, nonlinearity=nonlinearity)
        l_input = l_forward
    """
    In order to connect a recurrent layer to a dense layer, we need to
    flatten the first two dimensions (our "sample dimensions"); this will
    cause each time step of each sequence to be processed independently
    """
    l_shp = ReshapeLayer(l_forward, (-1, units[-1]))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=linear)
    l_out = ReshapeLayer(l_dense, (-1, seq_length, num_inputs))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    for layer in get_all_layers(l_out):
        sys.stderr.write("%s,%s\n" % (str(layer), str(layer.output_shape)))
    return l_out
