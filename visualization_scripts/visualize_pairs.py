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
from visualization_scripts.visualize_interpolate import resize


def init_wandb(H):
    tags = ["pairs", f"T{H.temp}|{H.temp_rest}"]

    if H.fixed:
        tags.append("fixed")

    wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/", tags=tags)
    wandb.config.update({"script": "vis_pairs"})

    if H.run_name:
        print(wandb.run.name)
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
    else:
        print(wandb.run.name)
        # wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
        wandb.run.name =  'PAIRS-' + wandb.run.name

def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/ffhq/all')
    parser.add_argument('--n_files', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--fixed', action="store_true")
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--temp_rest', type=float, default=0)

    # parser.add_argument('--size', type=int, default=128)
    # parser.add_argument('--n_steps', type=int, default=7)
    return parser


def visualize_pairs(H, file, vae, latent_ids, ls):
    pair = [2,3]

    l1, l2 = pair
    with torch.no_grad():
        z_dict = np.load(file)
        repr = {}
        repr['qm'] = [torch.tensor(z_dict[f'qm_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['pm'] = [torch.tensor(z_dict[f'pm_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['qv'] = [torch.tensor(z_dict[f'qv_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['pv'] = [torch.tensor(z_dict[f'pv_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['z'] = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]

        imgs = []
        for i in range(H.n_samples):
            if H.fixed:
                raise NotImplementedError()

            torch.random.manual_seed(i * 100)
            zs = repr["z"][:l1]
            # TODO: sample from prior here

            for j in range(H.n_samples):
                torch.random.manual_seed(j)

                temps = [0] * l2 + [H.temp] + [H.temp_rest] * (len(repr["z"]) - l2 - 1)
                img = vae.forward_samples_set_latents(1, zs, t=temps)

                imgs.append(img)

        imgs = [resize(img, size=(H.size, H.size)) for img in imgs]
        im = np.concatenate(imgs, axis=0).reshape((H.n_samples, H.n_samples, H.size, H.size, 3)).transpose(
            [0, 2, 1, 3, 4]).reshape([H.n_samples * H.size, H.n_samples * H.size, 3])
        i = file.split('/')[-1].split('.')[0]
        wandb.log({"samples": wandb.Image(im, caption=f"{i}")})

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)


    init_wandb(H)

    files = list(sorted(glob(os.path.join(H.latents_dir, "*.npz"))))[:H.n_files]
    for i, file in tqdm(enumerate(files)):

        visualize_pairs(H, file, ema_vae, latent_ids)

if __name__ == "__main__":
    main()
