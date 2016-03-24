from scipy.io import wavfile
import cPickle as pickle
import sys
import os
import numpy as np
import scipy

def fft(x):
    x_fft = scipy.fft(x)
    return np.hstack( (x_fft.real, x_fft.imag) )

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

offset_size = int(sys.argv[4])
print "offset_size: %i" % offset_size


if how_many_seconds > 0:
    sys.stderr.write("truncating data to %i seconds\n" % how_many_seconds)
    data = data[0: how_many_seconds*fs]

# e.g. 0 -> 0.8
train_data = data[ 0 : len(data)*valid_start ]
# e.g. 0.8 -> 0.9 
valid_data = data[ len(data)*valid_start : len(data)*test_start ]
# e.g. 0.9 ::
test_data = data[ len(data)*test_start :: ]

dd = [train_data, valid_data, test_data]

tmp_dirs = ["/tmp/dumps/train", "/tmp/dumps/valid", "/tmp/dumps/test"]
for tmp in tmp_dirs:
    os.makedirs(tmp)

num_saved = 0
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
    if offset_size > 0:
        offsets = range(0, x_size, x_size/offset_size)
    else:
        offsets = [0]

    for offset in offsets:
        seq = []
        b = 0
        while True:
            this_x = dd[i][ (b*x_size)+offset : ((b+1)*x_size)+offset ]
            if len(this_x) != x_size:
                break
            seq.append( fft(this_x) )
            if len(seq) == seq_length:
                #batches.append(seq)
		np.save( tmp_dirs[i] + ("/dump_%i_%i.npy" % (b, offset)), seq )
                num_saved += 1
                seq = []
            b += 1
    #dd[i] = np.asarray(batches, dtype="float32")
    #print "the shape of this array: %s" % (str(dd[i].shape))

#with open(sys.argv[4], "wb") as f:
#    pickle.dump( (dd, min_, max_), f, pickle.HIGHEST_PROTOCOL )

print "saved %i npy files: " % num_saved

"""

real_min = np.min(
    [np.min(dd[0][:,:,0:x_size]),
    np.min(dd[1][:,:,0:x_size]),
    np.min(dd[2][:,:,0:x_size])]
)
real_max = np.max(
    [np.max(dd[0][:,:,0:x_size]),
    np.max(dd[1][:,:,0:x_size]),
    np.max(dd[2][:,:,0:x_size])]
)

imag_min = np.min(
    [np.min(dd[0][:,:,x_size::]),
    np.min(dd[1][:,:,x_size::]),
    np.min(dd[2][:,:,x_size::])]
)
imag_max = np.max(
    [np.max(dd[0][:,:,x_size::]),
    np.max(dd[1][:,:,x_size::]),
    np.max(dd[2][:,:,x_size::])]
)

print "real min, real max, imag min, imag max = %f, %f, %f, %f" % (real_min, real_max, imag_min, imag_max)


# now compute the min/max
for i in range(0, len(dd)):

    print "normalising real elements..."
    dd[i][:,:,0:x_size] = (dd[i][:,:,0:x_size] - real_min) / (real_max - real_min)

    print "normalising imaginary elements..."
    dd[i][:,:,x_size::] = (dd[i][:,:,x_size::] - imag_max) / (imag_max - imag_min)

np.savez(sys.argv[5], dd[0], dd[1], dd[2], allow_pickle=False)

"""
