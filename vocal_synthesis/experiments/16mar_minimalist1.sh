#!/bin/bash

NAME=16mar_minimalist1
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/x1_50_all_use_mean.pkl ../models/${NAME}_use_mean.model > ../models/${NAME}_use_mean.txt
