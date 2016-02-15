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

def prepare(args):

    chunk_size = args["chunk_size"]

    l_in = InputLayer( (None, chunk_size) )
    l_hidden = DenseLayer(l_in, num_units=chunk_size*0.5, nonlinearity=leaky_rectify)
    #l_hidden2 = DenseLayer(l_hidden, num_units=chunk_size*0.5, nonlinearity=leaky_rectify)
    #l_hidden3 = DenseLayer(l_hidden2, num_units=chunk_size*0.75, nonlinearity=leaky_rectify)
    #l_out = DenseLayer(l_hidden3, num_units=chunk_size, nonlinearity=linear)
    l_out = DenseLayer(l_hidden, num_units=1, nonlinearity=linear)
    sys.stderr.write("Number of params in model: %i\n" % count_params(l_out))

    X = T.fmatrix('X')
    y = T.fmatrix('y')
    pred = lasagne.layers.get_output(l_out, X)
    get_out = theano.function([X], pred)

    params = lasagne.layers.get_all_params(l_out)
    loss = lasagne.objectives.squared_error(pred, y)
    loss = loss.mean()

    learning_rate = args["learning_rate"]
    momentum = args["momentum"]
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([X, y], loss, updates=updates)
    eval_fn = theano.function([X, y], loss)

    return {
        "l_out": l_out,
        "X": X,
        "y": y,
        "pred": pred,
        "get_out": get_out,
        "params": params,
        "loss": loss,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "updates": updates,
        "train_fn": train_fn,
        "eval_fn": eval_fn
    }

def train(args):

    chunk_size = args["chunk_size"]

    X_train, y_train = args["X_train"], args["y_train"]
    X_valid, y_valid = args["X_valid"], args["y_valid"]

    symbols = prepare(args)

    train_fn, eval_fn = symbols["train_fn"], symbols["eval_fn"]
    l_out = symbols["l_out"]

    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]

    print "epoch,train_loss,valid_loss,has_train_loss_improved,has_valid_loss_improved"
    
    best_val_score_so_far = float('inf')
    best_train_score_so_far = float('inf')
    best_model = None

    for epoch in range(0, num_epochs):
        b = 0
        train_losses = []
        while True:
            X_train_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_train_batch = y_train[b*batch_size : (b+1)*batch_size]
            train_losses.append( train_fn(X_train_batch, y_train_batch) )
            b += 1
            if b*batch_size >= len(X_train):
                break
            
        this_valid_loss = eval_fn(X_valid, y_valid)
        this_valid_has_improved = 0
        if this_valid_loss < best_val_score_so_far:
            best_val_score_so_far = this_valid_loss
            this_valid_has_improved = 1
            best_model = lasagne.layers.get_all_params_values(l_out)
            
        this_train_loss = np.mean(train_losses)
        this_train_has_improved = 0
        if this_train_loss < best_train_score_so_far:
            best_train_score_so_far = this_train_loss
            this_train_has_improved = 1

        print str(epoch) + "," \
            + str(this_train_loss) + "," \
            + str(this_valid_loss) + "," \
            + str(this_train_has_improved) + "," \
            + str(this_valid_has_improved)

    return best_model

def test(args):
    pass
