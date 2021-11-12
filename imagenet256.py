import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset


I256_DIR = "/scratch/s193223/celebahq2/CelebAMask-HQ/"

class ImageNet256Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=I256_DIR, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        meta = self.metadata.iloc[idx]
        img_name = os.path.join(self.img_dir, meta.file_name)
        X = Image.open(img_name)

        if self.transform :
            X = self.transform(X)

        return X, {}