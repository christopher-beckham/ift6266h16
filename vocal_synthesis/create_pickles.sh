#!/bin/bash

#for seq in 10 50 100 500 1000 2000 4000 8000; do
#    echo "creating pickle ${seq}..."
#    python create_pickles.py ${seq} -1 /Volumes/CB_RESEARCH/vocal_synthesis/${seq}.pkl
#done


for seq in 10 50 100 500 1000 2000 4000 8000; do
    echo "creating pickle ${seq}... for 60 seconds..."
    python create_pickles.py ${seq} 60 /Volumes/CB_RESEARCH/vocal_synthesis/${seq}_60sec.pkl
done