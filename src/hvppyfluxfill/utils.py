import torch
import PIL
import numpy as np
from PIL import Image, ImageDraw
from typing import Union
from diffusers.utils import load_image
from torchvision import transforms

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