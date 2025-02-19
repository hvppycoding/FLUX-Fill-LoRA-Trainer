import os
from typing import List, Optional
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from hvppyfluxfill.utils import full_mask, load_mask_image, prepare_mask_and_masked_image
from diffusers.utils import load_image
from logging import getLogger
from torchvision import transforms


logger = getLogger(__name__)


class DreamBoothDatasetWithMask(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        class_prompt: Optional[str]=None,
        class_data_root: Optional[str]=None,
        class_num: Optional[int]=None,
        size: int=1024,
        repeats: int=1,
        shuffle: bool=False,
        mask_suffix: str="_mask",
    ) -> None:
        self.size: int = size
        self.mask_suffix: str = mask_suffix
        self.repeats: int = repeats
        self.shuffle: bool = shuffle

        self.instance_data_root: str = instance_data_root
        self.instance_prompt: str = instance_prompt
        self.instance_image_tensors: List[torch.Tensor] = []
        self.instance_mask_tensors: List[PIL.Image.Image] = []
        self.instance_masked_image_tensors: List[torch.Tensor] = []
        self.custom_instance_prompts: List[str] = []
        self.initialize_instance_data()
        self.num_instance_images: int = len(self.instance_image_tensors)
        
        self.class_data_root: Optional[str] = class_data_root
        self.class_num: Optional[int] = class_num
        self.class_prompt: Optional[str] = class_prompt
        self.class_image_tensors: List[torch.Tensor] = []
        self.class_mask_tensors: List[PIL.Image.Image] = []
        self.class_masked_image_tensors: List[torch.Tensor] = []
        self.custom_class_prompts: List[str] = []
        if self.class_data_root is not None:
            self.initialize_class_data()
        self.num_class_images: int = len(self.class_image_tensors)
            
        self._length: int = max(self.num_instance_images, self.num_class_images)
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> dict:
        item = {}
        instance_index = index % self.num_instance_images
        item["instance_image_tensors"] = self.instance_image_tensors[instance_index]
        item["instance_mask_tensors"] = self.instance_mask_tensors[instance_index]
        item["instance_masked_image_tensors"] = self.instance_masked_image_tensors[instance_index]
        item["instance_prompts"] = self.custom_instance_prompts[instance_index]
        
        if self.num_class_images > 0:
            if self.shuffle:
                class_index = torch.randint(0, self.num_class_images, (1,)).item()
            else:
                class_index = index % self.num_class_images
            item["class_image_tensors"] = self.class_image_tensors[class_index]
            item["class_mask_tensors"] = self.class_mask_tensors[class_index]
            item["class_masked_image_tensors"] = self.class_masked_image_tensors[class_index]
            item["class_prompts"] = self.custom_class_prompts[class_index]
        return item
        
    def initialize_instance_data(self) -> None:
        if self.instance_data_root is None:
            raise ValueError("Instance images root is None.")
        if not Path(self.instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")
        
        filepaths = [path for path in list(Path(self.instance_data_root).iterdir())]
        filename2image = {}
        filename2mask = {}
        filename2prompt = {}
        
        for filepath in sorted(filepaths):
            filename_with_ext = filepath.name
            filename, ext = os.path.splitext(filename_with_ext)
            if filename.lower().endswith(self.mask_suffix):
                filename2mask[filename[:-len(self.mask_suffix)]] = filepath
            elif ext.lower() == ".txt":
                filename2prompt[filename] = filepath
            else:
                filename2image[filename] = filepath
                
        original_pil_images = []
        original_masks = []
        
        for filename in sorted(filename2image):
            try:
                image = load_image(str(filename2image[filename]))
                
                if filename in filename2mask:
                    mask_image = load_mask_image(str(filename2mask[filename]))
                else:
                    mask_image = full_mask(image.size)

                if filename in filename2prompt:
                    with open(filename2prompt[filename], "r") as prompt_file:
                        prompt_text = prompt_file.read()
                else:
                    prompt_text = self.instance_prompt

                original_pil_images.append(image)
                original_masks.append(mask_image)
                self.custom_instance_prompts.append(prompt_text)
                
            except (IOError, AttributeError, ValueError) as e:
                logger.warning(f"Failed to load image {filename}: {e}")
        
        train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(self.size)
        train_tensor = transforms.ToTensor()
        train_normalize = transforms.Normalize([0.5], [0.5])
        
        for image, mask in zip(original_pil_images, original_masks):
            resized_image = train_resize(image)
            resized_image = train_crop(resized_image)
            image_tensor = train_tensor(resized_image)
            image_tensor = train_normalize(image_tensor)
            
            resized_mask = train_resize(mask)
            resized_mask = train_crop(resized_mask)
            mask_tensor, masked_image_tensor = prepare_mask_and_masked_image(
                resized_image, resized_mask
            )
            
            self.instance_image_tensors.append(image_tensor)
            self.instance_mask_tensors.append(mask_tensor)
            self.instance_masked_image_tensors.append(masked_image_tensor)
        
        self.instance_image_tensors *= self.repeats
        self.instance_mask_tensors *= self.repeats
        self.instance_masked_image_tensors *= self.repeats
        self.custom_instance_prompts *= self.repeats
        
    def initialize_class_data(self) -> None:
        if self.class_data_root is None:
            raise ValueError("Class images root is None.")
        if not Path(self.class_data_root).exists():
            raise ValueError("Class images root doesn't exists.")
        
        filepaths = [path for path in list(Path(self.class_data_root).iterdir())]
        filename2image = {}
        filename2mask = {}
        filename2prompt = {}
        
        for filepath in sorted(filepaths):
            filename_with_ext = filepath.name
            filename, ext = os.path.splitext(filename_with_ext)
            if filename.lower().endswith(self.mask_suffix):
                filename2mask[filename[:-len(self.mask_suffix)]] = filepath
            elif ext.lower() == ".txt":
                filename2prompt[filename] = filepath
            else:
                filename2image[filename] = filepath
                
        original_pil_images = []
        original_masks = []
        
        for filename in sorted(filename2image):
            try:
                image = load_image(str(filename2image[filename]))
                
                if filename in filename2mask:
                    mask_image = load_mask_image(str(filename2mask[filename]))
                else:
                    mask_image = full_mask(image.size)

                if filename in filename2prompt:
                    with open(filename2prompt[filename], "r") as prompt_file:
                        prompt_text = prompt_file.read()
                else:
                    prompt_text = self.class_prompt

                original_pil_images.append(image)
                original_masks.append(mask_image)
                self.custom_class_prompts.append(prompt_text)
                
            except (IOError, AttributeError, ValueError) as e:
                logger.warning(f"Failed to load image {filename}: {e}")
        
        train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(self.size)
        train_tensor = transforms.ToTensor()
        # The values of mean and std for transform.normalize are not the desired mean and std, but rather the values to subtract and divide by, i.e., the estimated mean and std.
        # In your example you subtract 0.5 and then divide by 0.5 yielding an image with mean zero and values in range [-1, 1]
        train_normalize = transforms.Normalize([0.5], [0.5])

        
        for image, mask in zip(original_pil_images, original_masks):
            resized_image = train_resize(image)
            resized_image = train_crop(resized_image)
            image_tensor = train_tensor(resized_image)
            image_tensor = train_normalize(image_tensor)
            
            resized_mask = train_resize(mask)
            resized_mask = train_crop(resized_mask)
            mask_tensor, masked_image_tensor = prepare_mask_and_masked_image(
                resized_image, resized_mask
            )
            
            self.class_image_tensors.append(image_tensor)
            self.class_mask_tensors.append(mask_tensor)
            self.class_masked_image_tensors.append(masked_image_tensor)

def tensor_to_image(tensor: torch.Tensor) -> PIL.Image.Image:
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clip(0, 1)
    tensor = tensor.transpose(1, 2, 0)
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

def tensor_to_mask(mask: torch.Tensor) -> PIL.Image.Image:
    mask = mask.detach().cpu().numpy()
    mask = mask * 255
    mask = mask.clip(0, 255)
    mask = mask.astype(np.uint8)
    mask = mask[0]
    return Image.fromarray(mask, mode="L")