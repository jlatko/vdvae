from copy import copy
from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import imageio

from data import set_up_data
from latents import get_available_latents
from train_helpers import set_up_hyperparams, load_vaes
import wandb
wandb.init(project='vdvae_analysis', entity='johnnysummer', dir="/scratch/s193223/wandb/")


def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--attr_means_dir', type=str, default='/scratch/s193223/vdvae/attr_means/')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=7)
    parser.add_argument('--destination_dir', type=str, default='./visualizations/')
    return parser


def attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, fixed=True):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))


        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        for attr in attributes:
            batches = []
            for i in lv_points:
                zs_current = copy(zs)
                # get direction
                means_dict = np.load(os.path.join(H.attr_means_dir, f"{i}.npz"))
                direction = means_dict[f"{attr}_neg"] - means_dict[f"{attr}_pos"]

                for a in np.linspace(-1, 1, H.n_steps + 2):
                    zs_current[i] = zs[i] + a * direction
                    if fixed:
                        batches.append(ema_vae.forward_samples_set_latents(1, zs_current, t=0.1))
                    else:
                        batches.append(ema_vae.forward_samples_set_latents(1, zs_current[:i+1], t=0.1))

            n_rows = len(lv_points)
            im = np.concatenate(batches, axis=0).reshape((n_rows,  H.n_steps, *batches[0].shape[1:])).transpose(
                [0, 2, 1, 3, 4]).reshape([n_rows * batches[0].shape[1], batches[0].shape[2] * H.n_steps, 3])

            name_key = ""
            if fixed:
                name_key += "fixed_"
            fname = os.path.join(H.destination_dir, f"{attr}_{name_key}{idx}.png")
            imageio.imwrite(fname, im)

            wandb.log({f"{attr}_{name_key}": wandb.Image(im, caption=f"{attr}_{name_key}{idx}")})

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)

    if os.path.exists(H.destination_dir):
        if len(os.listdir(H.destination_dir)) > 0:
            print("WARNING: destination non-empty")
            sleep(5)
            print("continuing")
        #     raise RuntimeError('Destination non empty')
    else:
        os.makedirs(H.destination_dir)

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    print("Layer ids: ", np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * 65).astype(int)[1:-1])
    attributes = ["Young", "Male", "Smiling", "Wearing_Earrings", "Brown_Hair", "Blond_Hair", "Attractive"]
    for i in tqdm(range(H.n_samples)):
        idx = data_valid_or_test.metadata.iloc[i].idx
        attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, fixed=False)
        attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, fixed=True)


if __name__ == "__main__":
    main()
