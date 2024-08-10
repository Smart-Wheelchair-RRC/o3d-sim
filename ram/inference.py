'''
 * The Inference of RAM
 * Written by Xinyu Huang
'''
import torch

def inference_ram(image, model):

    with torch.no_grad():
        tags, tags_chinese = model.generate_tag(image)

    return tags[0],tags_chinese[0]


def inference_ram_openset(image, model):

    with torch.no_grad():
        tags = model.generate_tag_openset(image)

    return tags[0]
