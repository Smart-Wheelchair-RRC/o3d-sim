#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/ac26ZMwG7aT/sequence1/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/create_poses.py --dataset_dir "$DATASET_DIR"
    python scripts/rgbd_pointcloud.py --dataset_dir "$DATASET_DIR" --stride 1
    python scripts/gsam.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --box_threshold 0.32 --text_threshold 0.32 --version "v1"
    python scripts/main.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --version "v3"
done
