import numpy as np
import os
import pickle
import argparse
import random
from tqdm import tqdm

import cv2
import open3d as o3d
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/Datasets/mp3d_test/RPmz2sHmrrY/sequence4/",
    help="Directory for dataset",
)
parser.add_argument(
    "--gt_dir",
    type=str,
    default="/scratch/yash_mehan/mp3d_gt_instance_labelled/",
    help="Directory for GT data",
)
parser.add_argument(
    "--updated_gt_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/mp3d_gt_updated/",
    help="Directory for updated GT data",
)
parser.add_argument(
    "--version", type=str, default="output_objs_v1", help="Version string"
)

args = parser.parse_args()

dataset_dir = args.dataset_dir
version = args.version
output_dir = os.path.join(dataset_dir, f"output_{version}")
gt_dir = args.gt_dir
updated_gt_dir = args.updated_gt_dir

ALL_ROOM_TYPES = [
    "bathroom",
    "bedroom",
    "closet",
    "dining",
    "entryway",
    "family",
    "garage",
    "hallway",
    "library",
    "laundry",
    "kitchen",
    "living",
    "meeting",
    "lounge",
    "office",
    "porch",
    "recreation",
    "stairs",
    "toilet",
    "utility",
    "tv",
    "gym",
    "outdoor",
    "balcony",
    "other",
    "bar",
    "classroom",
    "sauna",
    "junk",
    "none",
]

OUR_ROOM_TYPES = [
    "bathroom",
    "bedroom",
    "closet",
    "dining",
    "entryway",
    "living",
    "garage",
    "hallway",
    "library",
    "laundry",
    "kitchen",
    "office",
    "porch",
    "stairs",
    "gym",
    "outdoor",
    "balcony",
    "bar",
    "other",
]

OUR_ROOM_DICT = {
    "bathroom": 0,
    "toilet": 0,
    "sauna": 0,
    "bedroom": 1,
    "closet": 2,
    "dining": 3,
    "entryway": 4,
    "family": 5,
    "tv": 5,
    "living": 5,
    "lounge": 5,
    "recreation": 5,
    "garage": 6,
    "hallway": 7,
    "library": 8,
    "laundry": 9,
    "utility": 9,
    "kitchen": 10,
    "meeting": 11,
    "office": 11,
    "porch": 12,
    "stairs": 13,
    "gym": 14,
    "outdoor": 15,
    "balcony": 16,
    "bar": 17,
    "other": 18,
    "classroom": 18,
    "junk": 18,
    "none": 18,
}

FINAL_ROOM_TYPES = [
    "bathroom",
    "living room",
    "dining room",
    "bedroom",
    "gym",
    "staircase",
    "kitchen",
    "hallway",
    "closet",
    "laundry room",
    "office",
    "game room",
    "utility room",
    "garage",
    "television room",
    "family room",
    "library",
    "conference auditorium",
    "lobby",
    "classroom",
    "lounge",
    "spa",
    "bar",
    "other",
]

FINAL_ROOM_DICT = {
    "bathroom": 0,
    "toilet": 0,
    "sauna": 21,
    "bedroom": 3,
    "closet": 8,
    "dining": 2,
    "entryway": 18,
    "family": 15,
    "tv": 14,
    "living": 1,
    "lounge": 20,
    "recreation": 11,
    "garage": 13,
    "hallway": 7,
    "library": 16,
    "laundry": 9,
    "utility": 12,
    "kitchen": 6,
    "meeting": 17,
    "office": 10,
    "porch": 23,
    "stairs": 5,
    "gym": 4,
    "outdoor": 23,
    "balcony": 23,
    "bar": 22,
    "other": 23,
    "classroom": 19,
    "junk": 23,
    "none": 23,
}


def generate_pastel_color():
    # generate (r, g, b) tuple of random numbers between 0.5 and 1, truncate to 2 decimal places
    r = round(random.uniform(0.4, 1), 2)
    g = round(random.uniform(0.4, 1), 2)
    b = round(random.uniform(0.4, 1), 2)
    color = np.array([r, g, b])
    return color


def load_gt_data(gt_dir, dataset_dir):
    dataset_name = dataset_dir.split("/")[-3]
    gt_file_name = f"{dataset_name}_xyz_rgb_o_r_inst.npy"
    gt = np.load(os.path.join(gt_dir, gt_file_name))  # (N, 9)
    return gt


def update_gt_data(gt):
    # Maps from old room_label_id to new room_label_id
    old_to_new_room_id = {
        ALL_ROOM_TYPES.index(room): FINAL_ROOM_DICT[room] for room in FINAL_ROOM_DICT
    }

    # Dictionary to track the new instance number for each unique (new_room_id, old_instance) pair
    unique_instance_mapping = {}

    # New instance number for each new room ID
    new_instance_counter = {i: 0 for i in range(0, len(FINAL_ROOM_TYPES))}

    for point in gt:
        old_room_id = int(point[7])
        old_instance = int(point[8])
        new_room_id = old_to_new_room_id.get(
            old_room_id, 23
        )  # Default to 'other' if not found

        # Unique key for the original room_id and instance combination
        unique_key = (new_room_id, old_room_id, old_instance)

        if unique_key not in unique_instance_mapping:
            unique_instance_mapping[unique_key] = new_instance_counter[new_room_id]
            new_instance_counter[new_room_id] += 1

        # Update the ground truth data with the new room_label_id and new instance number
        point[7] = new_room_id
        point[8] = unique_instance_mapping[unique_key]

    return gt


def save_updated_gt_data(gt, updated_gt_dir, dataset_dir):
    dataset_name = dataset_dir.split("/")[-3]
    gt_file_name = f"{dataset_name}_xyz_rgb_o_r_inst.npy"
    np.save(os.path.join(updated_gt_dir, gt_file_name), gt)
    return


def get_room_masks(gt):
    # gt shape is (N, 9) and format [x, y, z, r, g, b, obj, room_label_id, instance_number]
    unique_room_ids = np.unique(
        [
            "{}_{}".format(int(room_label), int(instance_num))
            for room_label, instance_num in gt[:, [7, 8]]
        ]
    )

    # Initialize a dictionary to store mask coordinates for each room
    room_masks = {}

    # Iterate over each unique room ID and plot
    for room_id in unique_room_ids:
        room_label, instance_num = map(int, room_id.split("_"))

        # Filter points belonging to the current room
        room_points = gt[(gt[:, 7] == room_label) & (gt[:, 8] == instance_num)]
        x, y, z = room_points[:, 0], room_points[:, 1], room_points[:, 2]
        room_masks[room_id] = (x, y, z)

    return room_masks


def get_kd_tree(room_masks):
    # Create a KD-tree for each room
    kd_trees = {}
    for room_id, (x, y, z) in room_masks.items():
        points = np.stack((x, y, z), axis=-1)
        kd_trees[room_id] = cKDTree(points)

    return kd_trees


def assign_room_to_obj(obj_nodes_dict, kd_trees):
    # Initialize a dictionary to store the room assignment for each object
    room_obj_assignments = {}

    # Iterate through each object
    for node_id, node_data in tqdm(obj_nodes_dict.items()):
        obj_bbox = node_data["bbox"]  # 8x3 array of bbox points

        # Initialize a dictionary to keep the sum of minimum distances for each room
        room_distance_sum = {room_id: 0 for room_id in kd_trees.keys()}

        # Iterate through the 8 points of the bounding box
        for point in obj_bbox:
            # Initialize a dictionary to keep the minimum distance for each room for this point
            room_min_distance = {}

            # Check the distance of the point to each room
            for room_id, kd_tree in kd_trees.items():
                distance, _ = kd_tree.query(
                    point[:3]
                )  # Consider x, y, z for 3D KD-tree
                room_min_distance[room_id] = distance

            # Add the minimum distance to the room's total distance sum
            for room_id, distance in room_min_distance.items():
                room_distance_sum[room_id] += distance

        # Assign the object to the room with the minimum sum of distances
        assigned_room = min(room_distance_sum, key=room_distance_sum.get)

        if assigned_room not in room_obj_assignments:
            room_obj_assignments[assigned_room] = []
        room_obj_assignments[assigned_room].append(node_id)

        node_data["room_id"] = assigned_room

        room_label_id = assigned_room.split("_")[0]
        node_data["room_label"] = FINAL_ROOM_TYPES[int(room_label_id)]

    return room_obj_assignments, obj_nodes_dict


def get_top30_objs_per_room(room_obj_assignments, obj_nodes_dict):
    room_obj_assignments_top30 = {}

    for room_id, obj_ids in room_obj_assignments.items():
        if len(obj_ids) <= 30:
            room_obj_assignments_top30[room_id] = obj_ids
        else:
            obj_points_counts = []

            for obj_id in obj_ids:
                pcd_path = obj_nodes_dict[obj_id]["pcd"]
                pcd = o3d.io.read_point_cloud(pcd_path)
                obj_points_counts.append((obj_id, len(pcd.points)))

            # Sort the list based on the number of points in descending order
            obj_points_counts.sort(key=lambda x: x[1], reverse=True)
            top_30_obj_ids = [obj_id for obj_id, _ in obj_points_counts[:30]]
            room_obj_assignments_top30[room_id] = top_30_obj_ids

    return room_obj_assignments_top30


def multi_scale_crop_image(image, bounding_box, kexp=0.1, levels=[0, 1, 2]):
    height, width = image.shape[:2]

    x, y, w, h = bounding_box
    xmin = int((x - w / 2).item() * width)
    ymin = int((y - h / 2).item() * height)
    xmax = int((x + w / 2).item() * width)
    ymax = int((y + h / 2).item() * height)

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


def save_roomwise_imgs(room_obj_assignments, obj_nodes_dict, img_dict):
    save_path = os.path.join(output_dir, "imgs_roomwise")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for room_id, node_ids in room_obj_assignments.items():
        room_path = os.path.join(save_path, f"{room_id}")
        if not os.path.exists(room_path):
            os.makedirs(room_path)

        for node_id in node_ids:
            k = 3
            points_contri = obj_nodes_dict[node_id]["points_contri"]
            source_ids = obj_nodes_dict[node_id]["source_ids"]
            sort = np.argsort(points_contri)[::-1]
            sorted_source_ids = np.array(source_ids)[sort][:k]

            all_crops = []
            processed_images = {}
            for source_id in sorted_source_ids:
                img_id, obj_id = source_id[0], source_id[1]

                # Read and preprocess the image only once per unique img_id
                if img_id not in processed_images:
                    img_path = img_dict[img_id]["img_path"]
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    processed_images[img_id] = img
                else:
                    img = processed_images[img_id]

                obj_bbox = img_dict[img_id]["objs"][int(obj_id)]["bbox"]
                cropped_imgs = multi_scale_crop_image(img, obj_bbox, levels=[2])
                all_crops.extend(cropped_imgs)

            # Save the cropped images
            for i, crop in enumerate(all_crops):
                cv2.imwrite(
                    os.path.join(room_path, f"{node_id}_{i}.jpg"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )

    return


def save_roomwise_pcds(room_obj_assignments, obj_nodes_dict):
    # generate colors equal to the length of room_labels
    room_id_to_color = {}
    for room_id in room_obj_assignments.keys():
        room_id_to_color[room_id] = generate_pastel_color()

    save_path = os.path.join(output_dir, "pcds_roomwise")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    room_pcds = {}

    # iterate over obj_nodes_dict and get the pcd for each node. Save the pcd with a color based on room label
    for node_id, node_data in obj_nodes_dict.items():
        obj_pcd_path = node_data["pcd"]
        obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
        obj_room_id = node_data["room_id"]

        color = room_id_to_color[obj_room_id]
        obj_pcd.paint_uniform_color(color)
        o3d.io.write_point_cloud(os.path.join(save_path, f"{node_id}.pcd"), obj_pcd)

        if obj_room_id not in room_pcds:
            room_pcds[obj_room_id] = []
        room_pcds[obj_room_id].append(obj_pcd)

    # room_pcds contains the list of pcds for each room label
    # we can merge them and save them in a separate folder
    save_path = os.path.join(output_dir, "pcds_roomwise_merged")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for room_id, pcd_list in room_pcds.items():
        merged_pcd = o3d.geometry.PointCloud()
        for pcd in pcd_list:
            merged_pcd += pcd
        o3d.io.write_point_cloud(os.path.join(save_path, f"{room_id}.pcd"), merged_pcd)

    return


def main():
    print(f"Working on dataset {dataset_dir} with version {version}...")

    # in output_dir, delete 3 dirs: imgs_roomwise, pcds_roomwise, pcds_roomwise_merged
    if os.path.exists(os.path.join(output_dir, "imgs_roomwise")):
        os.system(f"rm -r {os.path.join(output_dir, 'imgs_roomwise')}")
    if os.path.exists(os.path.join(output_dir, "pcds_roomwise")):
        os.system(f"rm -r {os.path.join(output_dir, 'pcds_roomwise')}")
    if os.path.exists(os.path.join(output_dir, "pcds_roomwise_merged")):
        os.system(f"rm -r {os.path.join(output_dir, 'pcds_roomwise_merged')}")

    print("Loading obj_nodes.pkl...")
    obj_nodes_path = os.path.join(output_dir, "scene_obj_nodes.pkl")
    with open(obj_nodes_path, "rb") as file:
        obj_nodes_dict = pickle.load(file)

    print("Loading img_dict.pkl...")
    img_dict_path = os.path.join(output_dir, "img_dict.pkl")
    with open(img_dict_path, "rb") as file:
        img_dict = pickle.load(file)

    gt = load_gt_data(gt_dir, dataset_dir)
    gt = update_gt_data(gt)
    if not os.path.exists(updated_gt_dir):
        os.makedirs(updated_gt_dir)
    # print("Saving updated gt data...")
    # save_updated_gt_data(gt, updated_gt_dir, dataset_dir)

    print("Getting room masks...")
    room_masks = get_room_masks(gt)

    print("Creating KD-trees for each room...")
    kd_trees = get_kd_tree(room_masks)

    print("Assigning rooms to objects...")
    room_obj_assignments, obj_nodes_dict = assign_room_to_obj(obj_nodes_dict, kd_trees)
    room_obj_assignments = get_top30_objs_per_room(room_obj_assignments, obj_nodes_dict)

    save_roomwise_pcds(room_obj_assignments, obj_nodes_dict)
    save_roomwise_imgs(room_obj_assignments, obj_nodes_dict, img_dict)

    # save obj_node_dict.pkl
    with open(obj_nodes_path, "wb") as file:
        pickle.dump(obj_nodes_dict, file)

    # save room_obj_assignments.pkl
    with open(os.path.join(output_dir, "room_obj_assignments.pkl"), "wb") as file:
        pickle.dump(room_obj_assignments, file)

    print("Done!")
    return


if __name__ == "__main__":
    main()

    # if os.path.exists(os.path.join(output_dir, "room_obj_assignments_v2.pkl")):
    #     os.rename(
    #         os.path.join(output_dir, "room_obj_assignments.pkl"),
    #         os.path.join(output_dir, "room_obj_assignments_v1.pkl"),
    #     )

    #     os.rename(
    #         os.path.join(output_dir, "room_obj_assignments_v2.pkl"),
    #         os.path.join(output_dir, "room_obj_assignments.pkl"),
    #     )

    # else:
    #     print(output_dir)
    #     print("room_obj_assignments_v2.pkl does not exist")
