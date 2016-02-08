import experiment
import numpy as np
from scipy.io import wavfile
import os
import sys
import cPickle as pickle

def generate_default(data, chunk_size):
    X_train = []
    y_train = []
    b = 0
    while True:
        this_chunk = data[b*chunk_size : (b+1)*chunk_size]
        if len(this_chunk) != chunk_size:
            break
        X_train.append( this_chunk[0 : len(this_chunk)-1] )
        y_train.append( [ this_chunk[len(this_chunk)-1] ] )
        b += 1
        if (b)*chunk_size >= len(data):
            break
    return np.asarray(X_train, dtype="float32"), np.asarray(y_train, dtype="float32")

data_folder = "/Volumes/CB_RESEARCH/vocal_synthesis/"
chunk_size = 200

fs, data = wavfile.read(data_folder + os.path.sep + "data.wav")
data = np.asarray(data, dtype="float32")
#mean = np.mean(data)
#std = np.std(data)
#data = (data - mean) / std
min_ = np.min(data)
max_ = np.max(data)
data = (data - min_) / (max_ - min_)
X_total, y_total = generate_default(data, chunk_size+1)

valid_size = 0.1
# ok, isolate out a validation set
idxs = [x for x in range(0, X_total.shape[0])]
train_idxs = idxs[ 0 : int(len(idxs)*(1-valid_size)) ]
valid_idxs = idxs[ int(len(idxs)*(1-valid_size)) :: ]
assert len(set(train_idxs).intersection(valid_idxs)) == 0

X_train, y_train = X_total[train_idxs], y_total[train_idxs]
X_valid, y_valid = X_total[valid_idxs], y_total[valid_idxs]

sys.stderr.write("X_train shape = %s\n" % str(X_train.shape))
sys.stderr.write("y_train shape = %s\n" % str(y_train.shape))
sys.stderr.write("X_valid shape = %s\n" % str(X_valid.shape))
sys.stderr.write("y_valid shape = %s\n" % str(y_valid.shape))

args = dict()
args["batch_size"] = 128
args["chunk_size"] = chunk_size
args["learning_rate"] = 0.01
args["momentum"] = 0.9
args["num_epochs"] = 20
args["X_train"] = X_train
args["y_train"] = y_train
args["X_valid"] = X_valid
args["y_valid"] = y_valid

model = experiment.train(args)

with open("out.model", "wb") as f:
    #pickle.dump({"params": model, "mean": mean, "std": std}, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump({"params": model, "min": min_, "max": max_}, f, pickle.HIGHEST_PROTOCOL)

