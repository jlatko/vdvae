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


def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--destination_dir', type=str, default='./visualizations/')
    return parser


def reconstruct_image(H, idx, ema_vae, latent_ids):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        print(lv_points)
        batches = []
        for i in lv_points:
            # create reconstructions using i first latents and low temperature (0.1)
            batches.append(ema_vae.forward_samples_set_latents(1, zs[:i], t=0.1))
        n_rows = len(batches)
        im = np.concatenate(batches, axis=0).reshape((n_rows, 1, *batches[0].shape[1:])).transpose(
            [0, 2, 1, 3, 4]).reshape([n_rows * batches[0].shape[1], batches[0].shape[2], 3])

        fname = os.path.join(H.destination_dir, f"{idx}.png")
        imageio.imwrite(fname, im)

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

    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)


if __name__ == "__main__":
    main()
