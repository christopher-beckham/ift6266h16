#!/bin/bash

NAME=run_basic_net_d3

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/100_60sec.pkl models/$NAME.model > models/$NAME.txt