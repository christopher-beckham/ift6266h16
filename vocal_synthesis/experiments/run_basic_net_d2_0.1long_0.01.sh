#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u run_basic_net_d2_0.1long_0.01.py $DATA_DIR/100_60sec.pkl models/basic_net_d2_0.1long_0.01.model > models/basic_net_d2_0.1long_0.01.txt
