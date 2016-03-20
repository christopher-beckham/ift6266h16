#!/bin/bash

NAME=19mar_8642468
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.fastmath=True \
        python -u $NAME.py $DATA_DIR/x0.05_5_all_use_mean.pkl \
	../../models/x0.05_5_all_use_mean/${NAME}_use_mean.model > ../../models/x0.05_5_all_use_mean/${NAME}_use_mean.txt
