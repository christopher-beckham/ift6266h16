from scipy.io import wavfile
import cPickle as pickle
import sys
import os
import numpy as np

test_start = 0.9
valid_start = 0.8

data_folder = os.environ["DATA_DIR"]

fs, data = wavfile.read(data_folder + os.path.sep + "data.wav")
data = np.asarray(data, dtype="float32")

# if x_length == 0.5, then
# x_length*fs = 16000*0.5 = 8000
# so an x^i represents 0.5 of a second
x_length = float(sys.argv[1])
print "each x will represent %f samples (%f seconds)" % (x_length*fs, x_length)
# a sequence is x_length*fs*seq_length long
seq_length = int(sys.argv[2])
print "each sequence will represent %f seconds" % (x_length*seq_length)

how_many_seconds = int(sys.argv[3]) # 60*20 = 20 minutes

if how_many_seconds > 0:
    sys.stderr.write("truncating data to %i seconds\n" % how_many_seconds)
    data = data[0: how_many_seconds*fs]

offset_size = int(sys.argv[4])

# e.g. 0 -> 0.8
train_data = data[ 0 : len(data)*valid_start ]
# e.g. 0.8 -> 0.9 
valid_data = data[ len(data)*valid_start : len(data)*test_start ]
# e.g. 0.9 ::
test_data = data[ len(data)*test_start :: ]

std_ = np.std(data)
print "standard deviation calculated: %f" % std_

train_data = (train_data) / std_
valid_data = (valid_data) / std_
test_data = (test_data) / std_

dd = [train_data, valid_data, test_data]

for i in range(0, len(dd)):
    x_size = int(x_length*fs)
    batches = []

    """
    This is to generate 10x more training data, e.g
    when offset=0, then we get things like
    data[0:8000], data[8000:16000], etc. and
    when offset=400 we get things like
    data[400:8400], data[8400:16400], etc.
    """
    for offset in range(0, x_size, x_size/offset_size):
        seq = []
        b = 0
        while True:
            if b*x_size >= dd[i].shape[0]:
                break
            this_x = dd[i][ (b*x_size)+offset : ((b+1)*x_size)+offset ]
            seq.append(this_x)
            if len(seq) == seq_length:
                batches.append(seq)
                seq = []
            b += 1
    dd[i] = np.asarray(batches, dtype="float32")
    print "the shape of this array: %s" % (str(dd[i].shape))

#with open(sys.argv[4], "wb") as f:
#    pickle.dump( (dd, min_, max_), f, pickle.HIGHEST_PROTOCOL )

np.savez(sys.argv[5], dd[0], dd[1], dd[2], allow_pickle=False)

np.savez(sys.argv[5] + ".sample", dd[0][0:100], dd[1][0:100], dd[2][0:100], allow_pickle=False)