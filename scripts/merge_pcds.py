import numpy as np
import os
import open3d as o3d
from tqdm import tqdm

dataset_dir = "/Users/kumaraditya/Desktop/Datasets/"
dataset_sequences = [
    "Pm6F8kyY3z2",
    "RPmz2sHmrrY",
    "ac26ZMwG7aT",
    "cV4RVeZvu5T",
    "e9zR4mvMWw7",
    "jh4fc5c5qoQ",
    "q9vSo1VnCiC",
    "sKLMLpTHeUy",
    "kEZ7cmS4wCh",
]

for sequence in dataset_sequences:
    print(f"Processing sequence: {sequence}")
    sequence_dir = os.path.join(dataset_dir, sequence)
    pcds_dir = os.path.join(sequence_dir, "pcds")
    pcds = []
    for filename in tqdm(os.listdir(pcds_dir)):
        if filename.endswith(".pcd"):
            pcd = o3d.io.read_point_cloud(os.path.join(pcds_dir, filename))
            pcds.append(pcd)

    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged_pcd += pcd

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.05)
    o3d.io.write_point_cloud(os.path.join(sequence_dir, "merged.pcd"), merged_pcd)
