import numpy as np
import cv2

import open3d as o3d
import faiss

import torch
import torch.nn.functional as F

from iou import compute_3d_iou_accuracte_batch, compute_3d_iou, compute_iou_batch
from utils import *


def merge_nodes(node1, node2, params, run_icp=False):
    # Merge source IDs: source_ids: [(img_id, obj_id), ...]
    source_ids = node1["source_ids"] + node2["source_ids"]
    count = len(source_ids)

    # Average the embeddings
    avg_clip_embed = (
        np.array(node1["clip_embed"]) * len(node1["source_ids"])
        + np.array(node2["clip_embed"]) * len(node2["source_ids"])
    ) / count

    avg_dino_embed = (
        np.array(node1["dino_embed"]) * len(node1["source_ids"])
        + np.array(node2["dino_embed"]) * len(node2["source_ids"])
    ) / count

    if run_icp:
        node2["pcd"] = vanilla_icp(node2["pcd"], node1["pcd"], params)

    # Combine point clouds
    merged_pcd = node1["pcd"]
    merged_pcd.points.extend(node2["pcd"].points)

    # make all points the same color (node1's color)
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(node1["pcd"].colors[0], (len(merged_pcd.points), 1))
    )
    merged_pcd = process_pcd(merged_pcd, params)

    bbox = get_bounding_box(merged_pcd, params)

    # Concatenate the points contributions from both nodes
    points_contri = node1["points_contri"] + node2["points_contri"]

    return {
        "source_ids": source_ids,
        "clip_embed": avg_clip_embed,
        "dino_embed": avg_dino_embed,
        "pcd": merged_pcd,
        "bbox": bbox,
        "points_contri": points_contri,
    }


def init_scene_nodes(img_data, img_id, params):
    # Initialize an empty dictionary to store scene object nodes
    scene_obj_nodes = {}

    # Retrieve the initial image data using the provided ID
    img_path = img_data["img_path"]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB format

    # Retrieve objects present in the image
    objs = img_data["objs"]
    node_count = 1

    for obj_id in objs.keys():
        obj_data = objs[obj_id]

        # Calculate similarities
        check_background_flag = check_background(obj_data)

        if check_background_flag:
            node_id = 0
            continue  # Skipping the background objects for now
        else:
            node_id = node_count
            node_count += 1

        color = generate_pastel_color()
        pcd = create_point_cloud(img_id, obj_data, color=color, params=params)
        pcd = process_pcd(pcd, params)

        bbox = get_bounding_box(pcd, params)
        if bbox.volume() < 1e-6 or len(pcd.points) < 10:
            continue

        if node_id not in scene_obj_nodes:
            # Store the object data in the scene object nodes dictionary
            scene_obj_nodes[node_id] = {
                "source_ids": [(img_id, obj_id)],
                "clip_embed": objs[obj_id]["clip_embed"],
                "dino_embed": objs[obj_id]["dino_embed"],
                "pcd": pcd,
                "bbox": bbox,
                "points_contri": [len(pcd.points)],
            }  # Count of points in the point cloud
        else:
            # Merge the object with the existing node
            scene_obj_nodes[node_id] = merge_nodes(
                scene_obj_nodes[node_id],
                {
                    "source_ids": [(img_id, obj_id)],
                    "clip_embed": objs[obj_id]["clip_embed"],
                    "dino_embed": objs[obj_id]["dino_embed"],
                    "pcd": pcd,
                    "bbox": bbox,
                    "points_contri": [len(pcd.points)],
                },
                params,
            )

    # print("Number of nodes in the scene: ", len(scene_obj_nodes))
    return scene_obj_nodes


def compute_overlap_matrix_2set(scene_obj_nodes, det_nodes, params):
    """
    Compute pairwise overlapping between two sets of objects in terms of point nearest neighbor.
    scene_obj_nodes is the existing objects in the scene, det_nodes is the new objects to be added to the scene
    """

    m = len(scene_obj_nodes)
    n = len(det_nodes)
    overlap_matrix = np.zeros((m, n))

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [
        np.asarray(obj["pcd"].points, dtype=np.float32)
        for obj in scene_obj_nodes.values()
    ]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map]  # m indices

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)

    points_new = [
        np.asarray(obj["pcd"].points, dtype=np.float32) for obj in det_nodes.values()
    ]

    # Assuming you can compute 3D IoU given the 'bbox' field in your dicts
    bbox_map_np = np.array(
        [obj["bbox"].get_box_points() for obj in scene_obj_nodes.values()]
    )
    bbox_map = torch.from_numpy(bbox_map_np).to(params["device"])

    bbox_new_np = np.array([obj["bbox"].get_box_points() for obj in det_nodes.values()])
    bbox_new = torch.from_numpy(bbox_new_np).to(params["device"])

    try:
        # Assuming you have a function called compute_3d_iou_accurate_batch that takes PyTorch tensors
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new)  # (m, n)
    except ValueError:
        # If you encounter the "Plane vertices are not coplanar" error, switch to axis-aligned bounding boxes
        bbox_map = []
        bbox_new = []

        for node in scene_obj_nodes.values():
            bbox_map.append(
                np.asarray(node["pcd"].get_axis_aligned_bounding_box().get_box_points())
            )

        for node in det_nodes.values():
            bbox_new.append(
                np.asarray(node["pcd"].get_axis_aligned_bounding_box().get_box_points())
            )

        bbox_map = torch.tensor(np.stack(bbox_map))
        bbox_new = torch.tensor(np.stack(bbox_new))

        # Assuming you have a function called compute_iou_batch that takes PyTorch tensors
        iou = compute_iou_batch(bbox_map, bbox_new)  # (m, n)

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i, j] < 1e-6:
                continue

            D, I = indices[i].search(
                points_new[j], 1
            )  # Search new object j in map object i

            overlap = (D < params["voxel_size"] ** 2).sum()  # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix


def compute_spatial_similarity(scene_obj_nodes, det_nodes, params):
    spatial_sim = compute_overlap_matrix_2set(scene_obj_nodes, det_nodes, params)
    spatial_sim = torch.from_numpy(spatial_sim).T
    spatial_sim = spatial_sim.to(params["device"])

    return spatial_sim


def compute_visual_similarity(scene_obj_nodes, det_nodes, params):
    """
    Compute the visual similarities between the detections and the objects.

    Args:
        scene_obj_nodes: a dict of N objects in the scene
        det_nodes: a dict of M detections
    Returns:
        A MxN tensor of visual similarities
    """

    # Extract clip embeddings from scene_obj_nodes and det_nodes dictionaries
    embed_type = params["embed_type"]
    obj_fts = np.array([obj[embed_type] for obj in scene_obj_nodes.values()])  # (N, D)
    det_fts = np.array([obj[embed_type] for obj in det_nodes.values()])  # (M, D)

    obj_fts = torch.from_numpy(obj_fts).to(params["device"])
    det_fts = torch.from_numpy(det_fts).to(params["device"])

    # Reshape tensors to match dimensions for cosine similarity
    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)

    # Compute cosine similarity
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)

    # Scale the visual similarity to be between 0 and 1
    # scaled_visual_sim = (visual_sim + 1) / 2

    return visual_sim


def compute_aggregate_similarity(scene_obj_nodes, det_nodes, params):
    """
    Compute the aggregate similarity between the detections and the objects.

    Args:
        scene_obj_nodes: a dict of N objects in the scene
        det_nodes: a dict of M detections
    Returns:
        A MxN tensor of aggregate similarities
    """

    spatial_sim = compute_spatial_similarity(scene_obj_nodes, det_nodes, params)
    visual_sim = compute_visual_similarity(scene_obj_nodes, det_nodes, params)
    aggregate_sim = (1 + params["alpha"]) * visual_sim + (
        1 - params["alpha"]
    ) * spatial_sim

    # if value in row is less than threshold, set it to -inf
    aggregate_sim[aggregate_sim < params["sim_threshold"]] = -float("inf")

    return aggregate_sim, spatial_sim, visual_sim


def update_scene_nodes(img_id, img_data, scene_obj_nodes, params):
    det_nodes = init_scene_nodes(
        img_data, img_id, params
    )  # Assuming img_data should be used

    if len(det_nodes) == 0:
        return scene_obj_nodes

    # Assuming you have a function named compute_aggregate_similarity to get aggregate similarity
    aggregate_sim, _, _ = compute_aggregate_similarity(
        scene_obj_nodes, det_nodes, params
    )

    # Initialize a new dictionary to store updated scene_obj_nodes
    updated_scene_obj_nodes = scene_obj_nodes.copy()

    # Find the maximum existing key in scene_obj_nodes
    max_scene_key = max(scene_obj_nodes.keys(), default=0)

    # Iterate through all detected nodes to merge them into existing scene_obj_nodes
    for i, det_key in enumerate(det_nodes.keys()):
        # If not matched to any object in the scene, add it as a new object
        if aggregate_sim[i].max() == float("-inf"):
            new_key = max_scene_key + det_key  # Create a new unique key
            updated_scene_obj_nodes[new_key] = det_nodes[det_key]
        else:
            # Merge with most similar existing object in the scene
            j = aggregate_sim[i].argmax().item()
            scene_key = list(scene_obj_nodes.keys())[j]
            matched_det = det_nodes[det_key]
            matched_obj = scene_obj_nodes[scene_key]

            # Merge the matched detection node into the matched scene node
            merged_obj = merge_nodes(matched_obj, matched_det, params)
            updated_scene_obj_nodes[scene_key] = merged_obj

    return updated_scene_obj_nodes


def filter_scene_nodes(scene_obj_nodes, params):
    print("Before filtering:", len(scene_obj_nodes))

    # Initialize a new dictionary to store the filtered scene_obj_nodes
    filtered_scene_obj_nodes = {}

    for key, obj in scene_obj_nodes.items():
        # Use len(obj['pcd'].points) to get the number of points and len(obj['source_ids']) to get the number of views
        if (
            len(obj["pcd"].points) >= params["obj_min_points"]
            and len(obj["source_ids"]) >= params["obj_min_detections"]
        ):
            filtered_scene_obj_nodes[key] = obj

    print("After filtering:", len(filtered_scene_obj_nodes))

    return filtered_scene_obj_nodes


def compute_overlap_matrix_nodes(params, scene_obj_nodes):
    n = len(scene_obj_nodes)
    overlap_matrix = np.zeros((n, n))

    point_arrays = [
        np.asarray(obj["pcd"].points, dtype=np.float32)
        for obj in scene_obj_nodes.values()
    ]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]

    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    for i, obj_i in enumerate(scene_obj_nodes.values()):
        for j, obj_j in enumerate(scene_obj_nodes.values()):
            if i != j:
                box_i = obj_i["bbox"]
                box_j = obj_j["bbox"]

                if params["merge_overlap_method"] == "iou":
                    iou = compute_3d_iou(box_i, box_j)
                    overlap_matrix[i, j] = iou

                elif params["merge_overlap_method"] == "max_overlap":
                    iou = compute_3d_iou(box_i, box_j, use_iou=False)
                    overlap_matrix[i, j] = iou

                elif params["merge_overlap_method"] == "nnratio":
                    iou = compute_3d_iou(box_i, box_j)
                    if iou == 0:
                        continue

                    D, I = indices[j].search(point_arrays[i], 1)
                    overlap = (D < params["voxel_size"] ** 2).sum()
                    overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix


def merge_overlap_nodes(params, scene_obj_nodes, overlap_matrix):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_keys = list(scene_obj_nodes.keys())
    merged_keys = set()  # Keep track of keys that have been merged into others

    for i, j, ratio in zip(x, y, overlap_ratio):
        key_i = kept_keys[i]
        key_j = kept_keys[j]

        # Skip if these keys have been merged into others
        if key_i in merged_keys or key_j in merged_keys:
            continue

        embed_type = params["embed_type"]
        visual_sim = F.cosine_similarity(
            torch.tensor(scene_obj_nodes[key_i][embed_type]).to(params["device"]),
            torch.tensor(scene_obj_nodes[key_j][embed_type]).to(params["device"]),
            dim=0,
        )

        overall_sim = (1 + params["alpha"]) * visual_sim + (1 - params["alpha"]) * ratio

        if overall_sim > params["merge_overall_thresh"]:
            if key_j in scene_obj_nodes:  # Check if key_j still exists
                scene_obj_nodes[key_j] = merge_nodes(
                    scene_obj_nodes[key_j],
                    scene_obj_nodes[key_i],
                    params,
                    run_icp=False,
                )
                del scene_obj_nodes[key_i]
                merged_keys.add(key_i)  # Mark key_i as merged

    return scene_obj_nodes


def merge_scene_nodes(scene_obj_nodes, params):
    if params["merge_overall_thresh"] > 0:
        print("Before merging:", len(scene_obj_nodes))

        # Compute the overlap matrix
        overlap_matrix = compute_overlap_matrix_nodes(params, scene_obj_nodes)

        # Merge overlapping nodes
        scene_obj_nodes = merge_overlap_nodes(params, scene_obj_nodes, overlap_matrix)

        print("After merging:", len(scene_obj_nodes))

    return scene_obj_nodes


def remove_empty_nodes(scene_obj_nodes, params):
    # Initialize a new dictionary to store the filtered scene_obj_nodes
    filtered_scene_obj_nodes = {}

    for key, obj in scene_obj_nodes.items():
        if len(obj["pcd"].points) > 10:
            filtered_scene_obj_nodes[key] = obj

    return filtered_scene_obj_nodes
