#!/bin/bash

NAME=19feb_testing_c

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/x0.1_50_all.pkl models/$NAME.model > models/$NAME.txt
