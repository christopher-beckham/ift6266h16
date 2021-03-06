#!/bin/bash

NAME=19mar_300x2
DATA=x0.125_10_all_offset10_std
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/$DATA.npy.npz \
	../../models/$DATA/${NAME}.model > ../../models/$DATA/${NAME}.txt
