import unittest
import os
import torch
from torch.utils.data import DataLoader
import shutil

from hvppyfluxfill.dataset import DreamBoothDatasetWithMask, tensor_to_image, tensor_to_mask

class TestDreamBoothDatasetWithMask(unittest.TestCase):
    def setUp(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.instance_data_root = os.path.join(self.current_dir, "train_data")
        self.class_data_root = os.path.join(self.current_dir, "class_data")
        
    def remake_output_dir(self, output_dir_name):
        output_dir_path = os.path.join(self.current_dir, output_dir_name)
        try:
            print(f"Removing {output_dir_name} folder")
            shutil.rmtree(output_dir_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error deleting folder: {e}")
        os.makedirs(output_dir_path, exist_ok=True)
        return output_dir_path

    def test_dataset(self):
        output_dir_path = self.remake_output_dir("output")
        
        dataset = DreamBoothDatasetWithMask(
            instance_data_root=self.instance_data_root,
            instance_prompt="A sks dog",
            class_data_root=self.class_data_root,
            class_prompt="A dog",
            shuffle=False
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                    for i in range(batch[key].shape[0]):
                        file_path = os.path.join(output_dir_path, f"{batch_idx}_{key}_{i}.png")
                        if batch[key][i].shape[0] == 1:
                            tensor_to_mask(batch[key][i]).save(file_path)
                        else:
                            tensor_to_image(batch[key][i]).save(file_path)
                else:
                    print(key, batch[key])

    def test_dataset_no_class_data(self):
        output_dir_path = self.remake_output_dir("output_no_class_data")
        
        dataset = DreamBoothDatasetWithMask(
            instance_data_root=self.instance_data_root,
            instance_prompt="A sks dog",
            shuffle=False
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                    for i in range(batch[key].shape[0]):
                        file_path = os.path.join(output_dir_path, f"{batch_idx}_{key}_{i}.png")
                        if batch[key][i].shape[0] == 1:
                            tensor_to_mask(batch[key][i]).save(file_path)
                        else:
                            tensor_to_image(batch[key][i]).save(file_path)
                else:
                    print(key, batch[key])
                    
if __name__ == '__main__':
    print(os.path.abspath(__file__))
    unittest.main()