import rnn_experiment as experiment
import numpy as np
from scipy.io import wavfile
import os
import sys
import cPickle as pickle

data_folder = os.environ["DATA_DIR"]

seq_length = int(sys.argv[1])
num_hidden = int(sys.argv[2])
use_lstm = True if sys.argv[3] == "lstm" else False

with open(data_folder + os.path.sep + ("%i.pkl" % seq_length) ) as f:
    dat = pickle.load(f)
X_train, X_valid, X_test = dat[0]

sys.stderr.write("X_train shape = %s\n" % str(X_train.shape))
sys.stderr.write("X_valid shape = %s\n" % str(X_valid.shape))
sys.stderr.write("X_test shape = %s\n" % str(X_test.shape))

args = dict()
args["num_inputs"] = 1
args["num_hidden_units"] = num_hidden
args["use_lstm"] = use_lstm
args["batch_size"] = 128
args["learning_rate"] = 0.01
args["momentum"] = 0.9
args["num_epochs"] = 5
args["X_train"] = X_train
args["X_valid"] = X_valid
args["X_test"] = X_test


model = experiment.train(args)

print "writing to file: s%i_h%i\n" % (seq_length, num_hidden)
with open(sys.argv[4], "wb") as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


