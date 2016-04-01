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

    sys.stderr.write("test...\n")
 
    out_nonlinearity=linear
    if "out_nonlinearity" in args:
	if args["out_nonlinearity"] == "sigmoid":
	    out_nonlinearity=lasagne.nonlinearities.sigmoid
            sys.stderr.write("using sigmout out nonlinearity\n")
        elif args["out_nonlinearity"] == "tanh":
	    out_nonlinearity=tanh
	    sys.stderr.write("using tanh out nonlinearity\n")

    if "forget_gate" in args:
        sys.stderr.write("using forget gate init with value = %i" % args["forget_gate"])
        forget_gate = Gate(b=Constant(args["forget_gate"]))
    else:
        forget_gate = Gate(b=Constant(0.0))

    l_input = InputLayer((None, None, num_inputs))
    l_noise = GaussianNoiseLayer(l_input, sigma=args["sigma"] if "sigma" in args else 0.0)
    l_prev = l_noise
    for unit in units:
    	l_forward = LSTMLayer(
            l_prev, num_units=unit, unroll_scan=False, precompute_input=True, nonlinearity=nonlinearity,
            forgetgate=forget_gate
        )
        l_prev = l_forward
    l_shp = ReshapeLayer(l_forward, (-1, units[-1]))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=out_nonlinearity)
    n_batch, n_time_steps, _ = l_input.input_var.shape
    l_out = ReshapeLayer(l_dense, (n_batch, n_time_steps, num_inputs))
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))
    for layer in get_all_layers(l_out):
        sys.stderr.write("%s,%s\n" % (str(layer), str(layer.output_shape)))
    return {"l_out": l_out, "l_in": l_input}
    #return l_out
