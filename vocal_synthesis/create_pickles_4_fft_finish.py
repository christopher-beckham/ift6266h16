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
j = 0
for i in range(0, len(tmp_dirs)):
    batches = []
    while True:
        filename = tmp_dirs[i] + ("/dump_%i.npy" % j)
        if os.path.isfile(filename):
            print "loading filename: %s" % filename
            batches.append( np.load(filename) )
            num_loaded += 1
            if x_size == -1:
                x_size = batches[-1].shape[1] / 2
                print "detected x size: %i" % x_size
            j += 1
        else:
            break
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

print "debug"
print dd[0][0]
print "end debug"

# now compute the min/max
for i in range(0, len(dd)):

    print "normalising real elements..."
    dd[i][:,:,0:x_size] = (dd[i][:,:,0:x_size] - real_min) / (real_max - real_min)

    print "normalising imaginary elements..."
    dd[i][:,:,x_size::] = (dd[i][:,:,x_size::] - imag_min) / (imag_max - imag_min)

print "debug"
print dd[0][0]
print "end debug"

np.savez(sys.argv[1], dd[0], dd[1], dd[2])
