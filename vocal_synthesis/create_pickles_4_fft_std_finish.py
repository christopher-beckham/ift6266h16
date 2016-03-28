import sys
import os
import numpy as np

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
    # this is what messes up the precision
    dd[i] = np.asarray(batches, dtype="float32")

std_ = np.std(
    [dd[0][:,:,:],
    dd[1][:,:,:],
    dd[2][:,:,:]]
)

print "num loaded: %i" % num_loaded

print "std = %f, %f" % std_

print "debug"
print dd[0][0]
print "end debug"

for i in range(0, len(dd)):

    print "normalising elements..."
    dd[i][:,:,:] = dd[i][:,:,:] / std_

print "debug"
print dd[0][0]
print "end debug"

np.savez(sys.argv[1], dd[0], dd[1], dd[2], allow_pickle=False)

np.savez(sys.argv[1] + ".sample", dd[0][0:100], dd[1][0:100], dd[2][0:100], allow_pickle=False)