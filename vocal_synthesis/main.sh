#for seq_length in 100 500 1000 2000 4000 8000; do
#for num_hidden in 10 20 50 100 200 500; do

#echo "seq length: $seq_length"
#echo "num hidden: $num_hidden"

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,nvcc.fastmath=True \
	python -u run_exp.py $DATA_DIR/100_60sec.pkl models/basic_net_d2.model > models/basic_net_d2.txt

#done
#done
