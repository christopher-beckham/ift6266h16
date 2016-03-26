#!/bin/bash

DATA=x0.5_6_all_offset2_fft
NAME=19mar_2000x1_sigmout
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/$DATA.npy.npz \
	../../models/$DATA/${NAME}.model > ../../models/$DATA/${NAME}.txt
