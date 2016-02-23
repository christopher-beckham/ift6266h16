#!/bin/bash

NAME=22feb_testing_a_rmsprop

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/x1_50_all.pkl ../models/$NAME.model > ../models/$NAME.txt
