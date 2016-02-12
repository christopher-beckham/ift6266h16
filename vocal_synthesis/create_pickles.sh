#!/bin/bash

for seq in 100 500 1000 2000 4000 8000; do
    echo "creating pickle ${seq}..."
    python create_pickles.py ${seq} /Volumes/CB_RESEARCH/vocal_synthesis/${seq}.pkl
done
