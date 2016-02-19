#!/bin/bash

NAME=basic_net_2000x1
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u $NAME.py $DATA_DIR/1000_60sec.pkl models/$NAME.model > models/$NAME.txt
