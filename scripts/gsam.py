import os
import sys
import pickle
import argparse
import gc
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as T

# Grounding DINO
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict

# Segment Anything
from segment_anything import build_sam, SamPredictor

# RAM
from ram.models import ram
from ram import inference_ram as inference

# Embeddings
import open_clip

from utils import *

# Parser
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
    default="/scratch/kumaraditya_gupta/Datasets/mp3d_train/sKLMLpTHeUy/sequence2/",
    help="Directory for dataset",
)
parser.add_argument("--stride", type=int, default=1, help="Stride value")
parser.add_argument(
    "--box_threshold", type=float, default=0.40, help="Box threshold value"
)
parser.add_argument(
    "--text_threshold", type=float, default=0.40, help="Text threshold value"
)
parser.add_argument("--version", type=str, default="v1", help="Version string")

args = parser.parse_args()

weights_dir = args.weights_dir
dataset_dir = args.dataset_dir
stride = args.stride
BOX_TRESHOLD = args.box_threshold
TEXT_TRESHOLD = args.text_threshold
version = args.version

background_prompt = "wall , door , doorframe , partition"
HOME = "/home/kumaraditya_gupta/instance-map"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_img_dict(img_dict, img_files, imgs_dir):
    print("Creating and initialzing image dictionary...")
    for i, img_file in enumerate(tqdm(img_files)):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(imgs_dir, img_file)

        # Add the img_path and ram_tags to the dictionary
        img_dict[img_id] = {
            "img_path": img_path,
            "ram_tags": background_prompt,
            "objs": {},
        }

    return img_dict


def generate_ram_tags(img_dict, img_files, imgs_dir):
    print("Loading RAM model...")
    image_size = 384  # default value
    pretrained = f"{weights_dir}/ram_swin_large_14m.pth"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    # Load model
    ram_model = ram(pretrained=pretrained, image_size=image_size, vit="swin_l")
    ram_model.eval()

    ram_model = ram_model.to(device)

    print("Generating RAM tags...")
    add_classes = ["other item"]
    remove_classes = [
        "room",
        "kitchen",
        "office",
        "house",
        "home",
        "building",
        "corner",
        "shadow",
        "carpet",
        "photo",
        "shade",
        "stall",
        "space",
        "aquarium",
        "apartment",
        "image",
        "city",
        "blue",
        "skylight",
        "hallway",
        "bureau",
        "modern",
        "salon",
        "doorway",
        "wall lamp",
        "pillar",
        "door",
        "basement",
        "workshop",
        "warehouse",
    ]
    bg_classes = ["wall", "floor", "ceiling", "office"]

    for i, img_file in enumerate(tqdm(img_files)):
        img_id = os.path.splitext(img_file)[0]

        # Only process images with valid poses
        pose_params = {"pose_dir": os.path.join(dataset_dir, "poses")}
        img_pose = get_pose(img_id, pose_params)
        if img_pose is None:
            continue

        img_path = os.path.join(imgs_dir, img_file)
        raw_image = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        image = transform(raw_image).unsqueeze(0).to(device)

        ram_tags = inference(image, ram_model)[0]
        ram_tags = ram_tags.split(" | ")  # Split the tags

        ram_tags = [tag for tag in ram_tags if tag not in bg_classes]
        for tag in ram_tags:
            words = tag.split()
            for word in words:
                if word in remove_classes:
                    ram_tags.remove(tag)
                    break
        ram_tags.extend(add_classes)

        # ram_tags = [tag.split() for tag in ram_tags] # Split the tags into words
        # ram_tags = [item for sublist in ram_tags for item in sublist] # Flatten the list
        # ram_tags = list(set(ram_tags) & set(req_tags)) # Get the intersection of the tags

        ram_tags = " , ".join(
            str(tag) for tag in ram_tags
        )  # Join the tags with a period
        # print(ram_tags)

        # Add the img_path and ram_tags to the dictionary
        img_dict[img_id] = {"img_path": img_path, "ram_tags": ram_tags, "objs": {}}

    del ram_model
    torch.cuda.empty_cache()
    gc.collect()

    return img_dict


def get_segment_models():
    CONFIG_PATH = (
        f"{HOME}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    # print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

    WEIGHTS_NAME_GD = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH_GD = f"{weights_dir}/{WEIGHTS_NAME_GD}"
    # print(WEIGHTS_PATH_GD, "; exist:", os.path.isfile(WEIGHTS_PATH_GD))

    # Load Model Grounding DINO
    GDINO_model = load_model(CONFIG_PATH, WEIGHTS_PATH_GD)

    sam_checkpoint_name = "sam_vit_h_4b8939.pth"
    sam_checkpoint = f"{weights_dir}/{sam_checkpoint_name}"
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    return GDINO_model, sam_predictor


def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy.to(device), image.shape[:2]
    )
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()


def generate_masks(img_dict, sam_predictor, GDINO_model):
    for img_id, img_data in tqdm(img_dict.items()):
        img_path = img_data["img_path"]

        TEXT_PROMPT = img_dict[img_id]["ram_tags"]
        image_source, image = load_image(img_path)

        # Get DINO Bounding Boxes
        boxes, logits, phrases = predict(
            model=GDINO_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )
        # print(boxes)

        if boxes.nelement() == 0:
            continue

        # Get SAM Masks
        sam_image = cv2.imread(img_path)
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        segmented_frame_masks = segment(sam_image, sam_predictor, boxes=boxes)

        num_objs = boxes.shape[0]
        objs = {}

        for j in range(num_objs):
            img_data["objs"][j] = {
                "bbox": boxes[j].cpu().numpy(),
                "phrase": phrases[j],
                "prob": logits[j].item(),
                "mask": segmented_frame_masks[j].cpu().squeeze().numpy(),
                "clip_embed": None,
                "dino_embed": None,
            }

    return img_dict


def crop_image(image, bounding_box):
    height, width = image.shape[:2]

    x, y, w, h = bounding_box
    xmin = int((x - w / 2).item() * width)
    ymin = int((y - h / 2).item() * height)
    xmax = int((x + w / 2).item() * width)
    ymax = int((y + h / 2).item() * height)
    cropped_image = image[ymin:ymax, xmin:xmax]

    return cropped_image


def resize_and_pad(image, desired_size=256):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    # find the larger dimension of the image
    max_dim = max(old_size)
    ratio = float(desired_size) / max_dim
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    # compute the deltas for padding
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # create a border around the image
    color = [0, 0, 0]  # black padding
    new_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_img


def load_clip_dino_models():
    # CLIP Model Loading and Preprocessing
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(device)

    # Load the pre-trained DINO model
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino_model = dino_model.to(device)
    dino_model.eval()

    # Define the image transformation pipeline
    transform_dino = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return clip_model, clip_preprocess, dino_model, transform_dino


def calc_dino_clip_embeddings(
    cropped_img, clip_model, clip_preprocess, dino_model, transform_dino
):
    # Load and preprocess the image for DinoV2
    dino_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    dino_image_tensor = (
        transform_dino(dino_img).unsqueeze(0).to(device)
    )  # Add batch dimension and send to device

    # Get the feature embedding from the model
    with torch.no_grad():
        dino_features = dino_model(dino_image_tensor)[0]

    # Load and preprocess the image for CLIP
    image_clip = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    input_clip = clip_preprocess(image_clip).unsqueeze(0).to(device)

    # Calculate CLIP embeddings
    with torch.no_grad():
        clip_features = clip_model.encode_image(input_clip)

    clip_features /= clip_features.norm(dim=-1, keepdim=True)

    return dino_features, clip_features


def store_embeddings(img_dict, clip_model, clip_preprocess, dino_model, transform_dino):
    for img_id, img_data in tqdm(img_dict.items()):
        img_path = img_data["img_path"]
        img = cv2.imread(img_path)

        # NOTE: Only for iPad
        # img = cv2.resize(img, (int(img.shape[1]/3.75), int(img.shape[0]/3.75)))

        if len(img_data["objs"]) == 0:
            continue

        for obj_id, obj_data in img_data["objs"].items():
            mask = obj_data["mask"]

            # If the mask isn't a numpy array, convert it to one
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()

            # Apply the mask to the image. This assumes your mask is binary (0s and 1s).
            # masked_image = img * np.expand_dims(mask, axis=2)
            cropped_img = crop_image(img, obj_data["bbox"])

            dino_features, clip_features = calc_dino_clip_embeddings(
                cropped_img, clip_model, clip_preprocess, dino_model, transform_dino
            )

            clip_features = clip_features.cpu().squeeze().numpy()
            dino_features = dino_features.cpu().squeeze().numpy()

            obj_data["clip_embed"] = clip_features
            obj_data["dino_embed"] = dino_features

    return img_dict


def main():
    print("Device: ", device)

    output_dir = os.path.join(dataset_dir, f"output_{version}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # # Load from pickle file
    # pickle_file_path = os.path.join(output_dir, "img_dict.pkl")
    # with open(pickle_file_path, "rb") as file:
    #     img_dict = pickle.load(file)

    imgs_dir = os.path.join(dataset_dir, "lowres_wide")
    img_files = [
        f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))
    ]
    # img_files = sorted(img_files, key=lambda x: int(x.split(".")[0]))
    img_files = img_files[::stride]
    print(f"Number of images: {len(img_files)}")

    img_dict = {}
    print("Initializing image dictionary...")
    # img_dict = init_img_dict(img_dict, img_files, imgs_dir)
    img_dict = generate_ram_tags(img_dict, img_files, imgs_dir)

    print("Loading models...")
    GDINO_model, sam_predictor = get_segment_models()

    print("Generating masks...")
    img_dict = generate_masks(img_dict, sam_predictor, GDINO_model)

    print("Loading CLIP and DINO models...")
    (
        clip_model,
        clip_preprocess,
        dino_model,
        transform_dino,
    ) = load_clip_dino_models()

    print("Calculating embeddings...")
    img_dict = store_embeddings(
        img_dict, clip_model, clip_preprocess, dino_model, transform_dino
    )

    print("Saving image dictionary...")
    pickle_file_path = os.path.join(output_dir, "img_dict.pkl")
    with open(pickle_file_path, "wb") as file:
        pickle.dump(img_dict, file)
    print("Done!")

    # del sam_predictor
    # del GDINO_model
    del clip_model
    del dino_model
    del img_dict
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
