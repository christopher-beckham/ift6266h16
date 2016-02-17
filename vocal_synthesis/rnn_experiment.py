import sys
import os
from scipy.io import wavfile
# ---
import numpy as np
import theano
from theano import tensor as T
# ---
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
# ---
from time import time
import imp
# ---
import cPickle as pickle

def prepare(args):

    sys.stderr.write("loading config: %s\n" % ("configurations/" + args["config"]))
    config = imp.load_source("config", "configurations/" + args["config"])

    l_out = config.get_net(args)

    if "in_model" in args:
        sys.stderr.write("loading existing model at: %s\n" % args["in_model"])
        layers = get_all_layers(l_out)[::-1]
        sys.stderr.write("debug: %s\n" % str(layers))
        with open(args["in_model"]) as f:
            in_model = pickle.load(f)
        for i in range(0, len(layers)):
            try:
                sys.stderr.write("debug: success\n")
                set_all_param_values(layers[i], in_model)
                break
            except:
                continue

    X_train = args["X_train"]

    if "seq_length" not in args:
        seq_length = X_train.shape[1]
    else:
        seq_length = args["seq_length"]

    X = T.tensor3('X')
    net_out = get_output(l_out, X)
    get_out = theano.function([X], net_out)

    loss = squared_error(net_out[:,0:seq_length-1,:], X[:,1::,:]).mean()
    params = get_all_params(l_out, trainable=True)
    loss_fn = theano.function([X], loss)

    learning_rate = args["learning_rate"]
    momentum = args["momentum"]
    if "adagrad" in args:
        updates = adagrad(loss, params, learning_rate)
    else:
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([X], loss, updates=updates)
    eval_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out)

    return {
        "l_out": l_out,
        "X": X,
        "get_out": get_out,
        "params": params,
        "loss": loss,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "updates": updates,
        "train_fn": train_fn,
        "eval_fn": eval_fn,
        "out_fn": out_fn
    }

def train(args):
    X_train, X_valid = args["X_train"], args["X_valid"]

    symbols = prepare(args)

    train_fn, eval_fn = symbols["train_fn"], symbols["eval_fn"]
    l_out = symbols["l_out"]

    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]

    print "epoch,train_loss,valid_loss,has_train_loss_improved,has_valid_loss_improved,duration"
    
    best_val_score_so_far = float('inf')
    best_train_score_so_far = float('inf')
    best_model = None

    for epoch in range(0, num_epochs):

        t0 = time()

        b = 0
        train_losses = []
        while True:
            X_train_batch = X_train[b*batch_size : (b+1)*batch_size]
            train_losses.append( train_fn(X_train_batch) )
            b += 1
            if b*batch_size >= len(X_train):
                break
            
        this_valid_loss = eval_fn(X_valid)
        this_valid_has_improved = 0
        if this_valid_loss < best_val_score_so_far:
            best_val_score_so_far = this_valid_loss
            this_valid_has_improved = 1
            best_model = lasagne.layers.get_all_param_values(l_out)
            
        this_train_loss = np.mean(train_losses)
        this_train_has_improved = 0
        if this_train_loss < best_train_score_so_far:
            best_train_score_so_far = this_train_loss
            this_train_has_improved = 1

        t1 = time() - t0

        print str(epoch) + "," \
            + str(this_train_loss) + "," \
            + str(this_valid_loss) + "," \
            + str(this_train_has_improved) + "," \
            + str(this_valid_has_improved) + "," \
            + str(t1)

    return best_model

def generate(args, model, seed, length):
    X_test = args["X_test"]
    symbols = prepare(args)

    out_fn = symbols["out_fn"]

    l_out = get_net(args)

    set_all_param_values(l_out, model)

    """
    seed = X_valid[0:1]
    seed = seed.reshape( (seed.shape[1], 1, 1) )
    seed = seed[0].reshape( (1,1,1) )
    """

    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = eval_fn(prev_input)
        generated_seq.append(next_input.flatten()[0])
        prev_input = next_input

    return generated_seq





