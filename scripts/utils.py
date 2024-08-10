import os
import numpy as np
import random
import cv2

from scipy.spatial.transform import Rotation as R

import open3d as o3d
from cuml.cluster import DBSCAN
import cupy as cp

from collections import Counter


def generate_pastel_color():
    # generate (r, g, b) tuple of random numbers between 0.5 and 1, truncate to 2 decimal places
    r = round(random.uniform(0.5, 1), 2)
    g = round(random.uniform(0.5, 1), 2)
    b = round(random.uniform(0.5, 1), 2)
    return (r, g, b)


"""
img_dict = {img_name: {img_path: str,
                        ram_tags: list_of_str,
                        objs: {0: {bbox: [x1, y1, x2, y2],
                                    phrase: str,
                                    clip_embed: [1, 1024]},
                                    dino_embed: [1, 1024]},
                                    mask: [h, w],
                                    prob: float,
                                    aabb: arr}
                                1: {...},
                        }
            img_name: {...},
            }
"""


def get_depth(img_name, params):
    # depth_path = os.path.join(depth_dir, img_name + '.npy')
    # depth = np.load(depth_path)

    depth_path = os.path.join(params["depth_dir"], img_name + ".png")
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32) / params["depth_scale"]
    return depth


def get_pose(img_name, params):
    pose_path = os.path.join(params["pose_dir"], img_name + ".txt")

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


def get_intrinsics(img_name, params):
    file_path = os.path.join(params["intrinsics_dir"], img_name + ".pincam")
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


def get_sim_cam_mat_with_fov(h, w, fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def get_realsense_cam_mat():
    K = np.array([[386.458, 0, 321.111], [0, 386.458, 241.595], [0, 0, 1]])
    return K


def get_kinect_cam_mat():
    K = np.array(
        [
            [9.7096624755859375e02, 0.0, 1.0272059326171875e03],
            [0.0, 9.7109600830078125e02, 7.7529718017578125e02],
            [0.0, 0.0, 1],
        ]
    )  # for wheel

    # K = np.array([[5.0449380493164062e+02, 0., 3.3090179443359375e+02],
    #               [0., 5.0470922851562500e+02, 3.1253039550781250e+02],
    #               [0., 0., 1.]]) # for handheld
    return K


def get_ipad_cam_mat():
    K = np.array(
        [
            [7.9555474853515625e02, 0.0, 3.6264770507812500e02],
            [0.0, 7.9555474853515625e02, 4.7412319946289062e02],
            [0.0, 0.0, 1.0],
        ]
    )

    # Downsample factor
    downsample_factor = 3.75

    # Scale down the camera matrix
    scaled_K = K.copy()
    scaled_K[0, 0] /= downsample_factor  # fx
    scaled_K[1, 1] /= downsample_factor  # fy
    scaled_K[0, 2] /= downsample_factor  # cx
    scaled_K[1, 2] /= downsample_factor  # cy

    return scaled_K


def get_iphone_cam_mat():
    K = np.array(
        [
            [6.6585675048828125e02, 0.0, 3.5704681396484375e02],
            [0.0, 6.6585675048828125e02, 4.8127374267578125e02],
            [0.0, 0.0, 1.0],
        ]
    )

    # Downsample factor
    downsample_factor = 3.75

    # Scale down the camera matrix
    scaled_K = K.copy()
    scaled_K[0, 0] /= downsample_factor  # fx
    scaled_K[1, 1] /= downsample_factor  # fy
    scaled_K[0, 2] /= downsample_factor  # cx
    scaled_K[1, 2] /= downsample_factor  # cy

    return scaled_K


def get_mp3d_cam_mat():
    K = np.array([[400.0, 0.0, 400.0], [0.0, 400.0, 400.0], [0.0, 0.0, 1.0]])
    return K


def create_point_cloud(img_id, obj_data, params, color=(1, 0, 0)):
    """
    Generates a point cloud from a depth image, camera intrinsics, mask, and pose.
    Only points within the mask and with valid depth are added to the cloud.
    Points are colored using the specified color.
    """
    depth = get_depth(img_id, params)
    pose = get_pose(img_id, params)
    mask = obj_data["mask"]

    if pose is None:
        return o3d.geometry.PointCloud()

    # Reproject the depth to 3D space
    rows, cols = np.where(mask)

    depth_values = depth[rows, cols]
    valid_depth_indices = (depth_values > 0) & (depth_values <= 10)

    rows = rows[valid_depth_indices]
    cols = cols[valid_depth_indices]
    depth_values = depth_values[valid_depth_indices]

    points2d = np.vstack([cols, rows, np.ones_like(rows)])

    if params["intrinsics_dir"] is not None:
        cam_mat = get_intrinsics(img_id, params)
    else:
        cam_mat = params["cam_mat"]
    cam_mat_inv = np.linalg.inv(cam_mat)
    points3d_cam = cam_mat_inv @ points2d * depth_values
    points3d_homo = np.vstack([points3d_cam, np.ones((1, points3d_cam.shape[1]))])

    # Parse the pose
    pos = np.array(pose[:3], dtype=float).reshape((3, 1))
    quat = pose[3:]
    rot = R.from_quat(quat).as_matrix()

    # # Apply rotation correction, to match the orientation z: backward, y: upward, and x: right
    # rot_ro_cam = np.eye(3)
    # rot_ro_cam[1, 1] = -1
    # rot_ro_cam[2, 2] = -1

    # rot = rot @ rot_ro_cam

    # # Apply position correction
    # pos[1] += params["cam_height"]

    # Create the pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rot
    pose_matrix[:3, 3] = pos.reshape(-1)

    points3d_global_homo = pose_matrix @ points3d_homo
    points3d_global = points3d_global_homo[:3, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d_global.T)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(color, (points3d_global.shape[1], 1))
    )

    # # Additional rotation to get the points in the correct orientation
    # new_rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    # new_rot_matrix = np.eye(4)
    # new_rot_matrix[:3, :3] = new_rot

    # # Apply new_rot to the point cloud using the transform function
    # pcd.transform(new_rot_matrix)

    return pcd


def fast_DBSCAN(point_cloud_o3d, eps=0.2, min_samples=20):
    if point_cloud_o3d.is_empty():
        return point_cloud_o3d

    # Convert Open3D point cloud to NumPy arrays
    points_np = np.asarray(point_cloud_o3d.points)
    colors_np = np.asarray(point_cloud_o3d.colors)

    # Convert NumPy array to CuPy array for GPU computations
    points_gpu = cp.asarray(points_np)

    # Create a DBSCAN instance with cuML
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the GPU data
    dbscan_model.fit(points_gpu)

    # Get the labels for the clusters
    labels_gpu = dbscan_model.labels_

    # Convert the labels back to a NumPy array
    labels = cp.asnumpy(labels_gpu)

    # Count the occurrence of each label to find the largest cluster
    label_counter = Counter(labels)
    label_counter.pop(-1, None)  # Remove the noise label (-1)
    if not label_counter:  # If all points are noise, return an empty point cloud
        return o3d.geometry.PointCloud()

    # Find the label of the largest cluster
    largest_cluster_label = max(label_counter, key=label_counter.get)

    # Filter the points and colors that belong to the largest cluster
    largest_cluster_points = points_np[labels == largest_cluster_label]
    largest_cluster_colors = colors_np[labels == largest_cluster_label]

    # Create a new Open3D point cloud with the points and colors of the largest cluster
    largest_cluster_point_cloud_o3d = o3d.geometry.PointCloud()
    largest_cluster_point_cloud_o3d.points = o3d.utility.Vector3dVector(
        largest_cluster_points
    )
    largest_cluster_point_cloud_o3d.colors = o3d.utility.Vector3dVector(
        largest_cluster_colors
    )

    return largest_cluster_point_cloud_o3d


def vanilla_icp(source, target, params):
    # Set ICP configuration
    config = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=params["icp_max_iter"]
    )

    icp_threshold = params["voxel_size"] * params["icp_threshold_multiplier"]

    # Run ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        config,
    )

    # Update pcd based on the transformation matrix obtained from ICP
    source.transform(result_icp.transformation)
    return source


def process_pcd(pcd, params, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=params["voxel_size"])

    if run_dbscan:
        pcd = fast_DBSCAN(pcd, eps=params["eps"], min_samples=params["min_samples"])

    return pcd


def get_bounding_box(pcd, params):
    try:
        return pcd.get_oriented_bounding_box(robust=True)
    except RuntimeError as e:
        # print(f"Met {e}, use axis aligned bounding box instead")
        return pcd.get_axis_aligned_bounding_box()


def check_background(obj_data):
    background_words = [
        "ceiling",
        # "wall",
        "floor",
        "pillar",
        "door",
        "basement",
        "room",
        "rooms",
        "workshop",
        "warehouse",
        "building",
        "apartment",
        "image",
        "city",
        "blue",
        "skylight",
        # "hallway",
        "bureau",
        "modern",
        "salon",
        "doorway",
        "house",
        "home",
        "carpet",
        "space",
        "exhaust",
    ]
    background_phrase = [
        "kitchen",
        "wall",
        "bedroom",
        "home office",
        "wood",
        "hardwood",
        "office",
        "bathroom",
        "living room",
    ]

    # background_words = ["room", "rooms"]
    # background_phrase = []

    obj_phrase = obj_data["phrase"]
    if obj_phrase in background_phrase:
        return True

    obj_words = obj_phrase.split()
    for word in obj_words:
        if word in background_words:
            return True
    return False
