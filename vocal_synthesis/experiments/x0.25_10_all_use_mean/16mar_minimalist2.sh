#!/bin/bash

NAME=16mar_minimalist2
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/x0.25_10_all_use_mean.pkl \
	../../models/x0.25_10_all_use_mean/${NAME}_use_mean.model > ../../models/x0.25_10_all_use_mean/${NAME}_use_mean.txt
