import argparse
import os
from time import sleep

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from collections import defaultdict
from tqdm import tqdm

from hps import Hyperparams

import numpy as np
import pandas as pd
from latents import get_latents, get_available_latents
import logging


def find_means(H, cols, layer_ind, latents_dir, handle_nan=False):
    z, meta = get_latents(latents_dir=latents_dir, layer_ind=layer_ind, splits=[1,2,3], allow_missing=False, handle_nan=handle_nan)
    logging.debug(z.shape)

    means_dict = {}

    for col in cols:
        y = np.array(meta[col] == 1)
        pos_mean = z[y].mean(axis=0)
        neg_mean = z[~y].mean(axis=0)
        means_dict[f"{col}_pos"] = pos_mean
        means_dict[f"{col}_neg"] = neg_mean

    np.savez(os.path.join(H.destination_dir, f"{layer_ind}.npz"), **means_dict)

def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--latents_dir', type=str, default="/scratch/s193223/vdvae/latents/")
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/attr_means/')
    parser.add_argument('--keys_set', type=str, default='full')
    parser.add_argument('--layer_ids_set', type=str, default='full')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--handle_nan', type=str, default=None)

    H.update(parser.parse_args(s).__dict__)
    return H

def setup(H):
    if H.keys_set == "small":
        cols = ["Young", "Male",  "Smiling", "Wearing_Earrings", "Brown_Hair"]
    elif H.keys_set == "big":
        cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby",
                "Straight_Hair", "Wavy_Hair", "Bangs",
                "Black_Hair", "Brown_Hair", "Blond_Hair",
                "Attractive",
                "Mouth_Slightly_Open",
                "Narrow_Eyes", "Bushy_Eyebrows",
                "Oval_Face", "Big_Lips", "Big_Nose", "Pointy_Nose",
                "Eyeglasses",
                "Heavy_Makeup", "Pale_Skin",
                "Wearing_Hat", "Wearing_Earrings", "Wearing_Lipstick"]
    elif H.keys_set == "full":
        cols = ['5_o_Clock_Shadow',
         'Arched_Eyebrows',
         'Attractive',
         'Bags_Under_Eyes',
         'Bald',
         'Bangs',
         'Big_Lips',
         'Big_Nose',
         'Black_Hair',
         'Blond_Hair',
         'Blurry',
         'Brown_Hair',
         'Bushy_Eyebrows',
         'Chubby',
         'Double_Chin',
         'Eyeglasses',
         'Goatee',
         'Gray_Hair',
         'Heavy_Makeup',
         'High_Cheekbones',
         'Male',
         'Mouth_Slightly_Open',
         'Mustache',
         'Narrow_Eyes',
         'No_Beard',
         'Oval_Face',
         'Pale_Skin',
         'Pointy_Nose',
         'Receding_Hairline',
         'Rosy_Cheeks',
         'Sideburns',
         'Smiling',
         'Straight_Hair',
         'Wavy_Hair',
         'Wearing_Earrings',
         'Wearing_Hat',
         'Wearing_Lipstick',
         'Wearing_Necklace',
         'Wearing_Necktie',
         'Young']

    else:
        raise ValueError(f"Unknown keys set {H.keys_set}")

    if H.layer_ids_set == "small":
        latent_ids = [0,1,2,3,5,10,15,20,30,40,50]
    elif H.layer_ids_set == "mid":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53, 58, 63]
    elif H.layer_ids_set == "full":
        latent_ids = get_available_latents()
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
    for i in tqdm(latent_ids):
        find_means(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan)


if __name__ == "__main__":
    main()