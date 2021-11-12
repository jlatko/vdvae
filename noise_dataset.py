import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset


CELEBAHQ_DIR = "/scratch/s193223/celebahq2/CelebAMask-HQ/"

def get_gaussian_noise(shape):
    return

def get_uniform_noise(shape):
    return

class NoiseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, resolution=256, length=1000, channels=3, noise_type="gaussian", **kwargs):
        self.resolution = resolution
        self.length = length
        self.noise_type = noise_type
        self.channels = channels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shape = (self.resolution, self.resolution, self.channels)

        if self.noise_type == "gaussian":
            X = torch.randn(shape)
        elif self.noise_type == "uniform":
            X = torch.rand(shape) * 2 - 1

        return X, {}