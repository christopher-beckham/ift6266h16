#!/bin/bash

NAME=19feb_testing

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/x0.1_50_1800.pkl models/$NAME.model > models/$NAME.txt
