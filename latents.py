import logging
import os
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

CELEBAHQ_DIR = "/scratch/s193223/celebahq2/CelebAMask-HQ/"


def get_available_latents(latents_dir="/scratch/s193223/vdvae/latents/"):
    fname = os.listdir(latents_dir)[0]
    keys = list(np.load(os.path.join(latents_dir, fname)))
    latent_ids = list(sorted(set(int(k.split("_")[1]) for k in keys)))
    return latent_ids

def get_latents(latents_dir, layer_ind, splits=(1,2,3), root_dir=CELEBAHQ_DIR, allow_missing=False, allow_nan=False):
    metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
    metadata = metadata[metadata.split.isin(splits)]

    z = np.load(os.path.join(latents_dir, f"{metadata.iloc[0].idx}.npz"))[f"z_{layer_ind}"]
    shape = [len(metadata)] + list(z.shape)
    latents = np.zeros(shape, dtype=np.float32)
    rows_found = []
    rows_missing = []
    i = 0
    # for _, row in tqdm(metadata.iterrows(),  total=metadata.shape[0]):
    for _, row in metadata.iterrows():
        try:
            z = np.load(os.path.join(latents_dir, f"{row.idx}.npz"))[f"z_{layer_ind}"].astype(np.float32)
            if not np.isfinite(z).all():
                logging.WARNING(f"{row.idx}: z_{layer_ind} contains NaN or inf")
                if allow_nan:
                    z = np.nan_to_num(z)
                else:
                    raise ValueError(f"{row.idx}: z_{layer_ind} contains NaN or inf")
            latents[i] = z
            rows_found.append(row)
            i += 1
        except (FileNotFoundError, EOFError, BadZipFile) as e:
            if allow_missing:
                rows_missing.append(row)
                pass
            else:
                raise e
    if len(rows_missing) > 0 and allow_missing:
        logging.warning(f"Missing {len(rows_missing)}/{len(metadata)} files")
        metadata = pd.DataFrame(rows_found)
        latents = latents[:len(metadata)]
    return latents, metadata
