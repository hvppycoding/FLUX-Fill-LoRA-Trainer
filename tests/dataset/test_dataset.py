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
        self.output_dir = os.path.join(self.current_dir, "output")
        try:
            print("Removing output folder")
            shutil.rmtree(self.output_dir)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error deleting folder: {e}")
        os.makedirs(self.output_dir, exist_ok=True)

    def test_dataset(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
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
                        file_path = os.path.join(self.output_dir, f"{batch_idx}_{key}_{i}.png")
                        if batch[key][i].shape[0] == 1:
                            tensor_to_mask(batch[key][i]).save(file_path)
                        else:
                            tensor_to_image(batch[key][i]).save(file_path)
                else:
                    print(key, batch[key])            
      
if __name__ == '__main__':
    print(os.path.abspath(__file__))
    unittest.main()