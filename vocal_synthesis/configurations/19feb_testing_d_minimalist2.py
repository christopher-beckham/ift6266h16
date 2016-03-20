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

    num_hidden_units = 100

    l_input = InputLayer((None, None, num_inputs))
    l_forward = LSTMLayer(l_input, num_units=num_hidden_units, unroll_scan=False, precompute_input=True)
    l_forward2 = LSTMLayer(l_forward, num_units=num_hidden_units, unroll_scan=False, precompute_input=True)
    l_forward3 = LSTMLayer(l_forward2, num_units=num_hidden_units, unroll_scan=False, precompute_input=True)

    """
    In order to connect a recurrent layer to a dense layer, we need to
    flatten the first two dimensions (our "sample dimensions"); this will
    cause each time step of each sequence to be processed independently
    """
    l_shp = ReshapeLayer(l_forward3, (-1, num_hidden_units))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=linear)
    n_batch, n_time_steps, _ = l_input.input_var.shape
    l_out = ReshapeLayer(l_dense, (n_batch, n_time_steps, num_inputs))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    return {"l_in": l_input, "l_out":l_out}
