#!/bin/bash

NAME=19mar_300x1
DATA=x0.25_10_all_offset5_fft_std
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/$DATA.npy.npz \
	../../models/$DATA/${NAME}.model > ../../models/$DATA/${NAME}.txt
