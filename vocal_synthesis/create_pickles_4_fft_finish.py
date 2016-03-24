from scipy.io import wavfile
import cPickle as pickle
import sys
import os
import numpy as np
import scipy
import glob

tmp_dirs = ["/tmp/dumps/train", "/tmp/dumps/valid", "/tmp/dumps/test"]

dd = [None, None, None]

x_size = -1
num_loaded = 0
for i in range(0, len(tmp_dirs)):
    batches = []
    for filename in glob.glob(tmp_dirs[i] + "/*.npy"):
        batches.append( np.load(filename) )
        num_loaded += 1
        if x_size == -1:
            x_size = batches[-1].shape[1] / 2
    dd[i] = np.asarray(batches, dtype="float32")

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

print "num loaded: %i" % num_loaded

print "real min, real max, imag min, imag max = %f, %f, %f, %f" % (real_min, real_max, imag_min, imag_max)


# now compute the min/max
for i in range(0, len(dd)):

    print "normalising real elements..."
    dd[i][:,:,0:x_size] = (dd[i][:,:,0:x_size] - real_min) / (real_max - real_min)

    print "normalising imaginary elements..."
    dd[i][:,:,x_size::] = (dd[i][:,:,x_size::] - imag_max) / (imag_max - imag_min)

np.savez(sys.argv[1], dd[0], dd[1], dd[2], allow_pickle=False)
