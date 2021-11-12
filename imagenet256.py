import os
from glob import glob

from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset


I256_DIR = "/scratch/s193223/inet/"

class ImageNet256Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=I256_DIR, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = list(glob(root_dir + "/*.JPEG"))
        if len(self.files) == 0:
            raise RuntimeError(f"No files matching {root_dir}/*.JPEG")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        img_name = self.files[idx]
        X = Image.open(img_name)

        if self.transform :
            X = self.transform(X)

        return X, {}