#!/bin/bash

if [ $HOSTNAME == "chris" ]; then 
    export DATA_DIR="/Volumes/CB_RESEARCH/vocal_synthesis/"
elif [ $HOSTNAME == "cuda2.local" ]; then
    export DATA_DIR=~/vocal_synthesis_data
elif [ $HOSTNAME == "cuda4.rdgi.polymtl.ca" ]; then
    export DATA_DIR=~/vocal_synthesis_data
elif [ $HOSTNAME == "cuda1" ]; then
    export DATA_DIR=~/vocal_synthesis_data
fi
