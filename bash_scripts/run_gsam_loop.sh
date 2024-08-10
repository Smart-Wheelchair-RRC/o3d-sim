#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/ac26ZMwG7aT/sequence1/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/gsam.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 2 --box_threshold 0.35 --text_threshold 0.35 --version "output_objs_v1"
done
