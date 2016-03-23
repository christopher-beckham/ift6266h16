#!/bin/bash

NAME=19mar_2000x1
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/x0.25_5_all_more.npy.npz \
	../../models/x0.25_5_all_more/${NAME}.model > ../../models/x0.25_5_all_more/${NAME}.txt
