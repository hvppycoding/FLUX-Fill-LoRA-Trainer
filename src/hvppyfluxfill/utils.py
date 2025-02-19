import torch
import PIL
import numpy as np
from PIL import Image, ImageDraw
from typing import Union
from diffusers.utils import load_image
from torchvision import transforms
import random

def full_mask(size):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, 0, size[0], size[1]), fill=255)
    return mask

def load_mask_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    return load_image(image).convert("L")

def prepare_mask_and_masked_image(image, mask):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.5], [0.5])(image)

    mask = transforms.ToTensor()(mask.convert("L"))
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    masked_image = image * (mask < 0.5)
    return mask, masked_image

# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask