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

    units = args["units"]

    if "nonlinearity" not in args:
        nonlinearity = tanh
    else:
        if args["nonlinearity"] == "relu":
            nonlinearity = rectify
            sys.stderr.write("using relu nonlinearity\n")
        elif args["nonlinearity"] == "sigmoid":
            nonlinearity = sigmoid
            sys.stderr.write("using sigmoid nonlinearity\n")

    l_input = InputLayer((None, None, num_inputs))
    l_prev = l_input
    for unit in units:
    	l_forward = LSTMLayer(
            l_prev, num_units=unit, unroll_scan=False, precompute_input=True, nonlinearity=nonlinearity
        )
        l_prev = l_forward
    l_shp = ReshapeLayer(l_forward, (-1, units[-1]))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=tanh)
    n_batch, n_time_steps, _ = l_input.input_var.shape
    l_out = ReshapeLayer(l_dense, (n_batch, n_time_steps, num_inputs))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    for layer in get_all_layers(l_out):
        sys.stderr.write("%s,%s\n" % (str(layer), str(layer.output_shape)))
    return {"l_out": l_out, "l_in": l_input}
    #return l_out
