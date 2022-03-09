from copy import copy
from glob import glob
from time import sleep

import numpy as np
import os
import torch
from tqdm import tqdm
import wandb

import imageio

from vdvae.data.data import set_up_data
from vdvae.latents import get_available_latents
from vdvae.train_helpers import set_up_hyperparams, load_vaes

def init_wandb(H):
    tags = ["sample"]

    wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/", tags=tags)
    wandb.config.update({"script": "vis_sample"})

    if H.run_name:
        print(wandb.run.name)
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
    else:
        print(wandb.run.name)
        # wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
        wandb.run.name =  'SAMPLE-' + wandb.run.name

def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--n_files', type=int, default=1)
    # parser.add_argument('--size', type=int, default=128)
    # parser.add_argument('--n_steps', type=int, default=7)
    return parser


def visualize(H, file, ema_vae, latent_ids):
    with torch.no_grad():
        z_dict = np.load(file)
        qm = [torch.tensor(z_dict[f'qm_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        qv = [torch.tensor(z_dict[f'qv_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        import IPython
        IPython.embed()

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    init_wandb(H)

    files = list(sorted(glob(os.path.join(H.latents_dir, "*.npz"))))[:H.n_files]
    for i, file in tqdm(enumerate(files)):

        visualize(H, file, ema_vae, latent_ids)

if __name__ == "__main__":
    main()
