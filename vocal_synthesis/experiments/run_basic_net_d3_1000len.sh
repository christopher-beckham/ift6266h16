#!/bin/bash

NAME=run_basic_net_d3_1000len

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/1000_60sec.pkl models/$NAME.model > models/$NAME.txt
