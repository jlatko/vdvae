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


def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=7)
    parser.add_argument('--destination_dir', type=str, default='./visualizations/')
    return parser


def reconstruct_image(H, idx, ema_vae, latent_ids):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        for t in [0.1, 0.5, 0.8, 1]:
            batches = []
            for i in lv_points:
                # create reconstructions using i first latents and low temperature (0.1)
                batches.append(ema_vae.forward_samples_set_latents(1, zs[:i], t=t))
            n_rows = len(batches)
            im = np.concatenate(batches, axis=0).reshape((n_rows, 1, *batches[0].shape[1:])).transpose(
                [0, 2, 1, 3, 4]).reshape([n_rows * batches[0].shape[1], batches[0].shape[2], 3])

            fname = os.path.join(H.destination_dir, f"{idx}_{str(t).replace('.','_')}.png")
            imageio.imwrite(fname, im)


def interpolation(H, idx, idx2, ema_vae, latent_ids, fixed=True, all_above=False):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
        z_dict2 = np.load(os.path.join(H.latents_dir, f"{idx2}.npz"))
        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        zs2 = [torch.tensor(z_dict2[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        batches = []
        for i in lv_points:
            zs_current = copy(zs)
            for a in np.linspace(0, 1, H.n_steps + 2)[1:-1]:
                if all_above:
                    for j in range(i):
                        zs_current[j] = (1-a) * zs[j] + a * zs2[j]
                zs_current[i] = (1 - a) * zs[i] + a * zs2[i]
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
        if all_above:
            name_key += "above_"
        fname = os.path.join(H.destination_dir, f"interpolation_{name_key}{idx}.png")
        imageio.imwrite(fname, im)


def swap(H, idx, idx2, ema_vae, latent_ids):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
        z_dict2 = np.load(os.path.join(H.latents_dir, f"{idx2}.npz"))
        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        zs2 = [torch.tensor(z_dict2[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        im = np.zeros(shape=(H.num_variables_visualize * 256, H.num_variables_visualize * 256, 3))
        for i in range(H.num_variables_visualize):
            for j in range(H.num_variables_visualize):
                zs_current = copy(zs)
                zs_current[lv_points[i]:lv_points[j]] = zs2[lv_points[i]:lv_points[j]]
                im[i*256 : (i+1)*256, j*256 : (j+1)*256] = ema_vae.forward_samples_set_latents(1, zs_current, t=0.1)

        fname = os.path.join(H.destination_dir, f"swap_{idx}.png")
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

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    for i in tqdm(range(H.n_samples)):
        idx = data_valid_or_test.metadata.iloc[i].idx
        idx2 = data_valid_or_test.metadata.iloc[i+1].idx
        print("Layer ids: ", np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * 65).astype(int)[1:-1])
        interpolation(H, idx, idx2, ema_vae, latent_ids, fixed=False, all_above=False)
        interpolation(H, idx, idx2, ema_vae, latent_ids, fixed=False, all_above=True)
        interpolation(H, idx, idx2, ema_vae, latent_ids, fixed=True, all_above=False)
        interpolation(H, idx, idx2, ema_vae, latent_ids, fixed=True, all_above=True)
        reconstruct_image(H, idx, ema_vae, latent_ids)


if __name__ == "__main__":
    main()
