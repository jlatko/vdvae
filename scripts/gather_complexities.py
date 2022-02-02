import io

import wandb
from PIL import Image
from skimage.filters.rank import entropy
from skimage.morphology import disk
from time import sleep

import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from vdvae.data.data import set_up_data
from vdvae.train_helpers import set_up_hyperparams, parse_hparams
import pandas as pd


def mean_local_entropy(x, radius=3):
    x = (x * 255).numpy().astype("uint8")
    entropies_per_channel = []
    for i in range(x.shape[0]):
        entropies_per_channel.append(
            np.mean(entropy(x[i], disk(radius)))
        )
    return np.mean(entropies_per_channel)

def get_size_bytesio(img, ext="JPEG", optimize=False):
    with io.BytesIO() as f:
        img.save(f, ext, optimize=optimize)
        s = f.getbuffer().nbytes
    return s

def compression(x, mode=0):
    x = (x * 255).numpy().astype("uint8")
    if x.shape[0] == 1:
        x = x[0]
    else:
        x = x.transpose(1,2,0)
    img = Image.fromarray(x)
    if mode == 0: # JPEG optimized/not-optimized
        optimized = get_size_bytesio(img, ext="JPEG", optimize=True)
        unoptimized = get_size_bytesio(img, ext="JPEG", optimize=False)
        return optimized / unoptimized

    if mode == 1: # JPEG optimized
        optimized = get_size_bytesio(img, ext="JPEG", optimize=True)
        return optimized

complexity_metrics = {
    "mean_local_entropy": mean_local_entropy,
    "compression": compression,
}

def get_complexities(H, data_valid, preprocess_fn):
    idx = -1
    all_stats = []
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=False, shuffle=False)):
        data_input, target = preprocess_fn(x)
        for i in range(data_input.shape[0]):
            stat_dict = {}

            if H.dataset == "celebahq":
                idx = x[1]["idx"][i].item()
            else:
                idx += 1

            stat_dict["idx"] = idx


            all_stats.append(stat_dict)
        if H.n is not None and len(all_stats) >= H.n:
            break
    all_stats = pd.DataFrame(all_stats)
    all_stats.to_pickle(os.path.join(H.destination_dir, f"complexity.pkl"))



def add_params(parser):
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/complexities/')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('-n', type=int, default=None)
    return parser

def main():
    H = parse_hparams()

    if H.run_name is not None:
        run_name = H.run_name
    else:
        run_name = f"COMPLEX_{H.dataset}"

    tags = ["complexities"]

    wandb.init(project='vdvae_analysis', entity='johnnysummer', dir="/scratch/s193223/wandb/", tags=tags, name=run_name)
    H.destination_dir = wandb.run.dir  # ???

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    if H.use_train:
        dataset = data_train
    else:
        dataset = data_valid_or_test

    get_complexities(H, dataset, preprocess_fn)

if __name__ == "__main__":
    main()
