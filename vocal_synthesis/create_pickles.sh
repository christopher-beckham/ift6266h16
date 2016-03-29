#!/bin/bash

#for seq in 10 50 100 500 1000 2000 4000 8000; do
#    echo "creating pickle ${seq}..."
#    python create_pickles.py ${seq} -1 /Volumes/CB_RESEARCH/vocal_synthesis/${seq}.pkl
#done


#for seq in 10 50 100 500 1000 2000 4000 8000; do
#    echo "creating pickle ${seq}... for 60 seconds..."
#    python create_pickles.py ${seq} 60 /Volumes/CB_RESEARCH/vocal_synthesis/${seq}_60sec.pkl
#done

#python create_pickles_2.py 0.5 20 1800 $DATA_DIR/x0.5_20_1800.pkl

#python create_pickles_2.py 0.1 50 1800 $DATA_DIR/x0.1_50_1800.pkl

# python create_pickles_2.py 0.1 50 -1 $DATA_DIR/x0.1_50_all.pkl

#python create_pickles_2.py 1 50 -1 $DATA_DIR/x1_50_all.pkl


#
#python create_pickles_3.py 1 50 -1 $DATA_DIR/x1_50_all_use_mean.pkl
#

# each xi represents 0.25 of a second
# we want 10 seconds of sequence
#python create_pickles_3.py 0.25 40 -1 $DATA_DIR/x0.25_10_all_use_mean.pkl

# each xi represents 0.1 of a second
# we want 5 seconds of sequence
#python create_pickles_3.py 0.1 50 -1 $DATA_DIR/x0.1_5_all_use_mean.pkl

# each xi represents 1 second
# we want 5 seconds of sequence
#python create_pickles_3.py 1 5 -1 $DATA_DIR/x1_5_all_use_mean.pkl

#python create_pickles_3.py 0.25 20 -1 $DATA_DIR/x0.25_5_all_use_mean.pkl



#python create_pickles_4.py 0.25 20 -1 $DATA_DIR/x0.25_5_all_more.npy



#python create_pickles_4_fft.py 0.125 20 -1 2 $DATA_DIR/x0.125_5_all_offset2_fft.npy
#python create_pickles_4_fft_finish.py $DATA_DIR/x0.125_5_all_offset2_fft.npy

#python create_pickles_4_fft.py 0.5 6 -1 2
#python create_pickles_4_fft_finish.py $DATA_DIR/x0.5_6_all_offset2_fft.npy

#python create_pickles_4_fft.py 0.5 12 -1 3
#python create_pickles_4_fft_finish.py $DATA_DIR/x0.5_12_all_offset3_fft.npy

#python create_pickles_4_with_offsets.py 0.5 10 -1 10 $DATA_DIR/x0.5_5_all_offset10.npy

#python create_pickles_4_with_offsets_use_std.py 0.5 10 -1 10 $DATA_DIR/x0.5_5_all_offset10_std.npy


#python  create_pickles_4_with_offsets_use_std.py 0.5 20 -1 10 $DATA_DIR/x0.5_10_all_offset10_std.npy

#python create_pickles_4_with_offsets_use_std.py 0.25 40 -1 10 $DATA_DIR/x0.25_10_all_offset10_std.npy

#python create_pickles_4_fft_std.py 0.5 12 -1 3
#python create_pickles_4_fft_std_finish.py $DATA_DIR/x0.5_12_all_offset3_fft_std.npy

#python create_pickles_4_fft_std.py 0.25 40 -1 4
#python create_pickles_4_fft_std_finish.py $DATA_DIR/x0.25_10_all_offset4_fft_std.npy

#python create_pickles_4_fft_std.py 0.25 40 -1 5
#python create_pickles_4_fft_std_finish.py $DATA_DIR/x0.25_10_all_offset5_fft_std.npy

python create_pickles_4_with_offsets_use_std.py 0.125 80 -1 10 $DATA_DIR/x0.125_10_all_offset10_std.npy
