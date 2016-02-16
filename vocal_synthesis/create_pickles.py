from scipy.io import wavfile
import cPickle as pickle
import sys
import os
import numpy as np

test_start = 0.9
valid_start = 0.8

seq_length = int(sys.argv[1])
how_many_seconds = int(sys.argv[2])

data_folder = os.environ["DATA_DIR"]

fs, data = wavfile.read(data_folder + os.path.sep + "data.wav")
data = np.asarray(data, dtype="float32")

if how_many_seconds > 0:
    sys.stderr.write("truncating data to %i seconds\n" % how_many_seconds)
    data = data[0: how_many_seconds*fs]

# e.g. 0 -> 0.8
train_data = data[ 0 : len(data)*valid_start ]
# e.g. 0.8 -> 0.9 
valid_data = data[ len(data)*valid_start : len(data)*test_start ]
# e.g. 0.9 ::
test_data = data[ len(data)*test_start :: ]

min_ = np.min(data)
max_ = np.max(data)
train_data = (train_data - min_) / (max_ - min_)
valid_data = (valid_data - min_) / (max_ - min_)
test_data = (test_data - min_) / (max_ - min_)

dd = [train_data, valid_data, test_data]

for i in range(0, len(dd)):
    b = 0
    seqs = []
    while True:
        this_seq = dd[i][b*seq_length : (b+1)*seq_length]
        if len(this_seq) != seq_length:
            break
        this_seq = this_seq.reshape( (seq_length, 1) )
        seqs.append(this_seq)
        b += 1
    seqs = np.asarray(seqs)
    dd[i] = seqs

with open(sys.argv[3], "wb") as f:
    pickle.dump( (dd, min_, max_), f, pickle.HIGHEST_PROTOCOL )
