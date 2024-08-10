import os
import numpy as np
from tqdm import tqdm
import pickle
import argparse

import cv2
from PIL import Image

import open3d as o3d
import open_clip

import torch

from utils import *
from scene_graph import *


parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument(
    "--weights_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/checkpoints",
    help="Directory for weights",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/Datasets/arkitscenes/ChallengeDevelopmentSet/42446588/",
    help="Directory for dataset",
)
parser.add_argument("--stride", type=int, default=1, help="Stride value")
parser.add_argument("--version", type=str, default="v2", help="Version string")

args = parser.parse_args()

weights_dir = args.weights_dir
dataset_dir = args.dataset_dir
version = args.version
output_dir = os.path.join(dataset_dir, f"output_{version}")

imgs_dir = os.path.join(dataset_dir, "lowres_wide_ordered")
depth_dir = os.path.join(dataset_dir, "lowres_depth_ordered")
pose_dir = os.path.join(dataset_dir, "poses_ordered")
intrinsics_dir = os.path.join(dataset_dir, "lowres_wide_intrinsics_ordered")

img_dict_dir = os.path.join(output_dir, "img_dict.pkl")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {
    "init_img_id": "001",  # initialize the scene with this image, overriden by the first image in img_dict
    "stride": args.stride,  # stride value for scene graph
    "depth_scale": 1000,  # depth scale for converting depth image to meters (std value: 1000.0, 655.35 for habitat)
    "depth_dir": depth_dir,  # directory containing depth images
    "pose_dir": pose_dir,  # directory containing pose files
    "intrinsics_dir": intrinsics_dir,  # directory containing intrinsics files
    "device": device,  # device to use for processing
    "voxel_size": 0.025,  # voxel size for downsampling point clouds (std value: 0.025, 0.05, 0.1)
    "eps": 0.075,  # eps for DBSCAN (std value: 0.075, 0.125, 0.25)
    "min_samples": 10,  # min_samples for DBSCAN (std value: 10)
    "embed_type": "dino_embed",  # embedding type to use for visual similarity
    "sim_threshold": 1.2,  # threshold for aggregate similarity while running update_scene_nodes (0.95, 1.2)
    "alpha": 0,  # weight for visual similarity while computing aggregate similarity
    "merge_overlap_method": "nnratio",  # metric to use for merging overlapping nodes
    "merge_overall_thresh": 1.2,  # threshold for overall similarity while merging nodes in scene (0.95, 1.2)
    "obj_min_points": 50,  # minimum number of points in a node while filtering scene nodes
    "obj_min_detections": 2,  # minimum number of detections in a node while filtering scene nodes
    "icp_threshold_multiplier": 1.5,  # threshold multiplier for ICP
    "icp_max_iter": 2000,  # maximum number of iterations for ICP
    "cam_mat": get_kinect_cam_mat(),  # camera matrix, get_sim_cam_mat_with_fov(900, 900, 90), overriden by intrinsics_dir
    "cam_height": 1.5,  # camera height
    "img_size": (256, 192),  # image size, (900, 900), (2048, 1536)
    "k_exp": 0.1,  # k_exp for multi-scale cropping
    "k_img": 2,  # number of images to consider for multi-scale cropping
    "bbox": "xyxy",  # bounding box format, xyxy, xywh
}


def crop_image(image, bounding_box, bbox="xywh"):
    height, width = image.shape[:2]

    if bbox == "xywh":
        x, y, w, h = bounding_box
        xmin = int((x - w / 2).item() * width)
        ymin = int((y - h / 2).item() * height)
        xmax = int((x + w / 2).item() * width)
        ymax = int((y + h / 2).item() * height)
    elif bbox == "xyxy":
        x1, y1, x2, y2 = bounding_box
        xmin = int(x1)
        ymin = int(y1)
        xmax = int(x2)
        ymax = int(y2)

    cropped_image = image[ymin:ymax, xmin:xmax]

    return cropped_image


def multi_scale_crop_image(
    image, bounding_box, kexp=0.1, levels=[0, 1, 2], bbox="xywh"
):
    height, width = image.shape[:2]

    if bbox == "xywh":
        x, y, w, h = bounding_box
        xmin = int((x - w / 2).item() * width)
        ymin = int((y - h / 2).item() * height)
        xmax = int((x + w / 2).item() * width)
        ymax = int((y + h / 2).item() * height)
    elif bbox == "xyxy":
        x1, y1, x2, y2 = bounding_box
        xmin = int(x1)
        ymin = int(y1)
        xmax = int(x2)
        ymax = int(y2)

    # Initialize list to hold cropped images
    cropped_images = []

    # Iterate over each level
    for l in levels:
        # Calculate new bounding box coordinates
        xl1 = max(0, xmin - (xmax - xmin) * kexp * l)
        yl1 = max(0, ymin - (ymax - ymin) * kexp * l)
        xl2 = min(xmax + (xmax - xmin) * kexp * l, width - 1)
        yl2 = min(ymax + (ymax - ymin) * kexp * l, height - 1)

        # Crop and append the image
        cropped_images.append(image[int(yl1) : int(yl2), int(xl1) : int(xl2)])

    return cropped_images


def calc_multi_scale_clip_embeds(
    img_dict, scene_obj_nodes, clip_model, clip_preprocess, params
):
    for node_id in tqdm(scene_obj_nodes.keys()):
        # For k highest points_contri, find the corresponding source_ids
        k = params["k_img"]
        points_contri = scene_obj_nodes[node_id]["points_contri"]
        source_ids = scene_obj_nodes[node_id]["source_ids"]

        sort = np.argsort(points_contri)[::-1]
        sorted_source_ids = np.array(source_ids)[sort][:k]

        # List to hold all crops from all top-k images
        all_crops = []

        # Process each source_id, avoid reading and processing the same image multiple times
        processed_images = {}
        for source_id in sorted_source_ids:
            img_id, obj_id = source_id[0], source_id[1]

            # Read and preprocess the image only once per unique img_id
            if img_id not in processed_images:
                img_path = img_dict[img_id]["img_path"]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB
                )  # Convert image from BGR to RGB format
                processed_images[img_id] = img
            else:
                img = processed_images[img_id]

            obj_bbox = img_dict[img_id]["objs"][int(obj_id)]["bbox"]
            cropped_imgs = multi_scale_crop_image(
                img,
                obj_bbox,
                kexp=params["k_exp"],
                levels=[0, 1, 2],
                bbox=params["bbox"],
            )
            all_crops.extend(cropped_imgs)

        # Convert all crops to CLIP-preprocessed tensors
        all_crops_tensors = [
            clip_preprocess(Image.fromarray(crop)).unsqueeze(0) for crop in all_crops
        ]
        all_crops_tensors = torch.cat(all_crops_tensors).to(device)

        # Get CLIP embeddings in one shot
        with torch.no_grad():
            all_clip_features = clip_model.encode_image(all_crops_tensors)
            all_clip_features /= all_clip_features.norm(dim=-1, keepdim=True)

        avg_clip_features = all_clip_features.mean(dim=0)
        avg_clip_features /= avg_clip_features.norm()

        scene_obj_nodes[node_id]["multi_clip_embed"] = avg_clip_features.cpu().numpy()

    return scene_obj_nodes


def main():
    with open(img_dict_dir, "rb") as file:
        img_dict = pickle.load(file)

    img_dict = {
        k: v for i, (k, v) in enumerate(img_dict.items()) if i % params["stride"] == 0
    }

    for img_id in img_dict.keys():
        params["init_img_id"] = img_id
        scene_obj_nodes = init_scene_nodes(
            img_dict[params["init_img_id"]], params["init_img_id"], params
        )

        if len(scene_obj_nodes) > 0:
            break

    if len(scene_obj_nodes) == 0:
        print("No objects detected in the scene")
        return
    # # choose init_img_id as the first image in the img_dict
    # params["init_img_id"] = list(img_dict.keys())[0]

    # scene_obj_nodes = init_scene_nodes(
    #     img_dict[params["init_img_id"]], params["init_img_id"], params
    # )

    print("Number of nodes in the scene: ", len(scene_obj_nodes))

    counter = 0

    for img_id, img_data in tqdm(img_dict.items()):
        if len(img_data["objs"]) == 0 or img_id == params["init_img_id"]:
            continue

        scene_obj_nodes = update_scene_nodes(img_id, img_data, scene_obj_nodes, params)
        scene_obj_nodes = remove_empty_nodes(scene_obj_nodes, params)

        counter += 1
        if counter % 25 == 0:
            scene_obj_nodes = merge_scene_nodes(scene_obj_nodes, params)
            scene_obj_nodes = remove_empty_nodes(scene_obj_nodes, params)

    scene_obj_nodes = merge_scene_nodes(scene_obj_nodes, params)
    scene_obj_nodes = filter_scene_nodes(scene_obj_nodes, params)

    save_pcd_dir = os.path.join(output_dir, "pcds")
    if not os.path.exists(save_pcd_dir):
        os.makedirs(save_pcd_dir)

    print("Number of objs in the scene: ", len(scene_obj_nodes))
    for node_id, node_data in scene_obj_nodes.items():
        pcd_path = os.path.join(save_pcd_dir, f"{node_id}.pcd")
        o3d.io.write_point_cloud(pcd_path, node_data["pcd"])
        node_data["pcd"] = pcd_path  # Replace the Open3D object with the file path
        node_data["bbox"] = np.array(
            node_data["bbox"].get_box_points()
        )  # Convert the Open3D bounding box to a NumPy array

    # Calc multi-scale CLIP Embedss
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(device)

    scene_obj_nodes = calc_multi_scale_clip_embeds(
        img_dict, scene_obj_nodes, clip_model, clip_preprocess, params
    )

    # Save the modified dictionary using pickle
    scene_obj_nodes_dir = os.path.join(output_dir, "scene_obj_nodes.pkl")

    # if this file exists, rename the old filr to scene_obj_nodes_v1.pkl
    if os.path.exists(scene_obj_nodes_dir):
        os.rename(
            scene_obj_nodes_dir, os.path.join(output_dir, "scene_obj_nodes_v1.pkl")
        )

    with open(scene_obj_nodes_dir, "wb") as f:
        pickle.dump(scene_obj_nodes, f)


if __name__ == "__main__":
    main()
