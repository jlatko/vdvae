import os
import numpy as np
import pandas as pd
from tqdm import tqdm

CELEBAHQ_DIR = "/scratch/s193223/celebahq2/CelebAMask-HQ/"


def get_available_latents(latents_dir="/scratch/s193223/vdvae/latents/"):
    fname = os.listdir(latents_dir)[0]
    keys = list(np.load(os.path.join(latents_dir, fname)))
    latent_ids = list(sorted(set(int(k.split("_")) for k in keys)))
    return latent_ids

def get_latents(latents_dir, layer_ind, splits=(1,2,3), root_dir=CELEBAHQ_DIR):
    metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
    metadata = metadata[metadata.split.isin(splits)]

    z = np.load(os.path.join(latents_dir, f"{metadata.iloc[0].idx}.npz"))[f"z_{layer_ind}"]
    shape = [len(metadata)] + list(z.shape)
    print(shape)
    latents = np.zeros(shape, dtype=np.float32)
    for i, (_, row) in tqdm(enumerate(metadata.iterrows())):
        z = np.load(os.path.join(latents_dir, f"{row.idx}.npz"))[f"z_{layer_ind}"].astype(np.float32)
        latents[i] = z

    return latents, metadata
