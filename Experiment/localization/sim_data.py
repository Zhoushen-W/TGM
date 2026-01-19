import os
import cv2
import torch
from torch.utils.data import Dataset


class sim_dataset(Dataset):
    def __init__(self, data_path, data_type, num_dir=None, img_channel=1):
        self.data_type = data_type
        if self.data_type == "dem":
            assert num_dir is not None, "when data_type is dem, num_dir should be given"
            self.img_dir = os.path.join(data_path, f"{self.data_type}_images", num_dir)
            self.meta_dir = os.path.join(data_path, f"{self.data_type}_meta", num_dir)
        elif self.data_type == "beam":
            self.img_dir = os.path.join(data_path, f"{self.data_type}_images")
            self.meta_dir = os.path.join(data_path, f"{self.data_type}_meta")
        else:
            raise ValueError(f"data_type should be beam or dem, but got {self.data_type}")
        self.names = sorted(
            [f.replace(".png", "") for f in os.listdir(self.img_dir) if f.endswith(".png")]
        )
        self.img_channel = img_channel
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, f"{name}.png")
        # shape: [c, h, w]
        if self.img_channel == 1:
            img_tensor = torch.from_numpy(cv2.imread(img_path, 0) / 255.0).float()
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            img_tensor = torch.from_numpy(cv2.imread(img_path) / 255.0).float()
            img_tensor = img_tensor.permute(2, 0, 1)

        meta = torch.load(
            os.path.join(self.meta_dir, f"{name}.pt"),
            map_location="cpu",
            weights_only=True    
        )
        if self.data_type == "beam":
            return {"img": img_tensor, "prob": meta["prob"]}
        else:
            return {"img": img_tensor, "coord": meta["coord"]}
