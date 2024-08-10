# import open3d as o3d

import os
import gc
import numpy as np
import random
import cv2
from tqdm import tqdm
import pickle
from collections import Counter

from scipy.spatial.transform import Rotation as R

import open3d as o3d

import argparse


def get_pose(img_name, pose_dir):
    pose_path = os.path.join(pose_dir, img_name + ".txt")

    # check if the pose file exists, if it doesn't, return None
    # [x, y, z, qx, qy, qz, qw]
    if not os.path.exists(pose_path):
        return None

    with open(pose_path, "r") as f:
        pose = f.read().split()
        pose = np.array(pose).astype(np.float32)

        # change pose from [x, y, z, qw, qx, qy, qz] to [x, y, z, qx, qy, qz, qw]
        # pose = np.concatenate((pose[:3], pose[4:], pose[3:4]))
    return pose


def get_intrinsics(img_name, intrinsics_dir):
    file_path = os.path.join(intrinsics_dir, img_name + ".pincam")
    with open(file_path, "r") as file:
        line = file.readline().strip()

    # Extract parameters from the line
    (
        width,
        height,
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
    ) = map(float, line.split())

    # Construct the camera intrinsic matrix K
    K = np.array(
        [
            [focal_length_x, 0, principal_point_x],
            [0, focal_length_y, principal_point_y],
            [0, 0, 1],
        ]
    )

    return K


def create_pcd_from_rgbd(
    img_files_list, imgs_dir, depth_dir, pose_dir, intrinsics_dir=None
):
    pcd_global = o3d.geometry.PointCloud()

    # For each image, load the RGB-D image and transform the point cloud to the global frame
    for i, img_file in enumerate(tqdm(img_files_list)):
        # Load RGB and Depth images
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(imgs_dir, img_file)
        rgb_image = o3d.io.read_image(img_path)

        depth_path = os.path.join(depth_dir, img_id + ".png")
        depth_image = o3d.io.read_image(depth_path)

        # Load the camera intrinsics
        if intrinsics_dir is not None:
            cam_mat = get_intrinsics(img_id, intrinsics_dir)

        pose = get_pose(img_id, pose_dir)
        if pose is None:
            continue

        # intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #     900,  # width
        #     900,  # height
        #     450.0,  # fx
        #     450.0,  # fy
        #     450.0,  # cx
        #     450.0,  # cy
        # )

        # intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #     2048,  # width
        #     1536,  # height
        #     9.7096624755859375e02,  # fx
        #     9.7109600830078125e02,  # fy
        #     1.0272059326171875e03,  # cx
        #     7.7529718017578125e02,  # cy
        # )

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            256,
            192,
            cam_mat[0, 0],
            cam_mat[1, 1],
            cam_mat[0, 2],
            cam_mat[1, 2],
        )

        # Generate point cloud from RGB-D image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image,
            depth_image,
            depth_scale=1000,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )
        # o3d.visualization.draw_geometries([rgbd_image])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        voxel_size = 0.05  # adjust this value to change the number of points
        pcd = pcd.voxel_down_sample(voxel_size)

        # Parse the pose [x, y, z, qx, qy, qz, qw]
        pos = np.array(pose[:3], dtype=float).reshape((3, 1))
        quat = pose[3:]
        rot = R.from_quat(quat).as_matrix()

        # # Apply rotation correction, to match the orientation z: backward, y: upward, and x: right
        # rot_ro_cam = np.eye(3)
        # rot_ro_cam[1, 1] = -1
        # rot_ro_cam[2, 2] = -1

        # combined_rot = rot @ rot_ro_cam

        # cam_height = 1.50
        # pos[1] += cam_height

        # Create the pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot
        pose_matrix[:3, 3] = pos.reshape(-1)
        final_pose_matrix = pose_matrix

        # # Additional rotation to get the points in the correct orientation
        # new_rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        # new_rot_matrix = np.eye(4)
        # new_rot_matrix[:3, :3] = new_rot

        # # Apply the pose to the point cloud using the transform function
        # final_pose_matrix = new_rot_matrix @ pose_matrix

        pcd.transform(final_pose_matrix)

        # # Apply new_rot to the point cloud using the transform function
        # pcd.transform(new_rot_matrix)

        pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Add the point cloud to the global point cloud
        pcd_global += pcd

    pcd_global = pcd_global.voxel_down_sample(voxel_size=0.01)
    return pcd_global


def main(args):

    dataset_path = args.dataset_dir

    imgs_dir = os.path.join(dataset_path, "lowres_wide/")
    depth_dir = os.path.join(dataset_path, "lowres_depth/")
    pose_dir = os.path.join(dataset_path, "poses/")
    intrinsics_dir = os.path.join(dataset_path, "lowres_wide_intrinsics/")
    # save_dir = os.path.join(dataset_path, "output_v1/")
    save_dir = dataset_path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_files_list = [
        f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))
    ]
    # img_files_list = sorted(img_files_list, key=lambda x: int(x.split(".")[0]))
    img_files_list = sorted(img_files_list, key=lambda x: int(x.split("_")[0]))

    stride = args.stride
    img_files_list = img_files_list[::stride]

    pcd_global = create_pcd_from_rgbd(
        img_files_list, imgs_dir, depth_dir, pose_dir, intrinsics_dir
    )

    o3d.io.write_point_cloud(
        os.path.join(save_dir, "pointcloud_aligned.ply"), pcd_global
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script parameters")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/scratch/kumaraditya_gupta/Datasets/arkitscenes/ChallengeDevelopmentSet/42445677",
        help="Directory for dataset",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride value")

    args = parser.parse_args()
    main(args)
