import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset


CELEBAHQ_DIR = f"{BASE_DIR}/celebahq2/CelebAMask-HQ/"

class CelebAHQDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=CELEBAHQ_DIR, train=True, transform=None, splits=None, resolution=256):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.root_dir = root_dir
        self.resolution = resolution
        if resolution == 256:
            self.img_dir = os.path.join(self.root_dir, "img256")
        elif resolution == 1024:
            self.img_dir = os.path.join(self.root_dir, "CelebA-HQ-img")
        else:
            raise ValueError('resolution not supported')
        self.transform = transform
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.split_mapping = {
            True: [0],
            False: [1,2,3]
        }
        if splits is None:
            splits = self.split_mapping[train]
        if isinstance(splits, str):
            splits = [int(x) for x in splits.split(',')]
        self.metadata = self.metadata[self.metadata.split.isin(splits)]

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

        # filter out meta
        # meta = meta[~meta.isin(['idx', 'orig_idx', 'orig_file', 'split', 'index', 'file_name'])]
        # maybe not filter out to keep track of ids while saving

        return X, meta.to_dict()