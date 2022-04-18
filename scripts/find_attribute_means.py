import argparse
import os
from time import sleep

from vdvae.attributes import get_attributes

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from collections import defaultdict
from tqdm import tqdm

from vdvae.hps import Hyperparams
from vdvae.constants import BASE_DIR

import numpy as np
from vdvae.latents import get_latents, get_available_latents
import logging

# TODO: consider accounting for the most frequent and correlated attributes (earings only for females etc)
MIN_FREQ = 10

def get_means(z, meta, col):
    y = np.array(meta[col] == 1)
    pos_mean = z[y].mean(axis=0)
    neg_mean = z[~y].mean(axis=0)
    return pos_mean, neg_mean

def find_means(H, cols, layer_ind, latents_dir, handle_nan=False):
    z, meta = get_latents(latents_dir=latents_dir, layer_ind=layer_ind, splits=[0,1,2,3], allow_missing=False, handle_nan=handle_nan, key=H.latent_key)
    logging.debug(z.shape)

    means_dict = {}

    for col in cols:
        male = meta["Male"] == 1
        query = meta[col] == 1

        male_and_query = male & query
        male_and_not_query = male & (~query)
        female_and_query = (~male) & query
        female_and_not_query = (~male) & (~query)
        # logging.info(f"{col}:"
        #              f"\n male (y/n) {male_and_query.sum()} | {male_and_not_query.sum()} "
        #              f"\n female (y/n) {female_and_query.sum()} | {female_and_not_query.sum()} ")
        # male
        if male_and_query.sum() >= MIN_FREQ and male_and_not_query.sum() >= MIN_FREQ:
            pos_mean_male, neg_mean_male = get_means(z[male], meta[male], col)
            means_dict[f"{col}_pos_male"] = pos_mean_male
            means_dict[f"{col}_neg_male"] = neg_mean_male
            means_dict[f"{col}_diff_male"] = pos_mean_male - neg_mean_male
            means_dict[f"{col}_diff_grouped"] = means_dict[f"{col}_diff_male"]
        else:
            logging.info(f"{col} skipping male")
        # female
        if female_and_query.sum() >= MIN_FREQ and female_and_not_query.sum() >= MIN_FREQ:
            pos_mean_female, neg_mean_female = get_means(z[~male], meta[~male], col)
            means_dict[f"{col}_pos_female"] = pos_mean_female
            means_dict[f"{col}_neg_female"] = neg_mean_female
            means_dict[f"{col}_diff_female"] = pos_mean_female - neg_mean_female

            if f"{col}_diff_grouped" in means_dict:
                means_dict[f"{col}_diff_grouped"] = (means_dict[f"{col}_diff_male"] + means_dict[f"{col}_diff_female"]) / 2
            else:
                means_dict[f"{col}_diff_grouped"] = means_dict[f"{col}_diff_female"]
        else:
            logging.info(f"{col} skipping female")


        pos_mean, neg_mean = get_means(z, meta, col)
        means_dict[f"{col}_pos"] = pos_mean
        means_dict[f"{col}_neg"] = neg_mean
        means_dict[f"{col}_diff"] = pos_mean - neg_mean

    fname =  f"{layer_ind}.npz"
    np.savez(os.path.join(H.destination_dir, fname), **means_dict)

def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--latents_dir', type=str, default=f"{BASE_DIR}/vdvae/latents/")
    parser.add_argument('--destination_dir', type=str, default=f'{BASE_DIR}/vdvae/attr_means/')
    parser.add_argument('--keys_set', type=str, default='full')
    parser.add_argument('--layer_ids_set', type=str, default='full')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--handle_nan', type=str, default=None)
    parser.add_argument('--latent_key', type=str, default="z")
    # parser.add_argument('--group', action="store_true")

    H.update(parser.parse_args(s).__dict__)
    return H

def setup(H):
    cols = get_attributes(H.keys_set)

    if H.layer_ids_set == "small":
        latent_ids = [0,1,2,3,5,10,15,20,30,40,50]
    elif H.layer_ids_set == "mid":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53, 58, 63]
    elif H.layer_ids_set == "mid_mem":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53]
    elif H.layer_ids_set == "full":
        latent_ids = get_available_latents(latents_dir=H.latents_dir)
    else:
        raise ValueError(f"Unknown latent ids set {H.layer_ids_set}")

    logging.basicConfig(level=H.log_level)

    return cols, latent_ids

def main():
    H = parse_args()
    cols, latent_ids = setup(H)

    if os.path.exists(H.destination_dir):
        if len(os.listdir(H.destination_dir)) > 0:
            print("WARNING: destination non-empty")
            sleep(5)
            print("continuing")
        #     raise RuntimeError('Destination non empty')
    else:
        os.makedirs(H.destination_dir)

    logging.info(cols)
    logging.info(latent_ids)

    scores = defaultdict(list)
    for i in tqdm(list(reversed(latent_ids))):
        find_means(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan)


if __name__ == "__main__":
    main()