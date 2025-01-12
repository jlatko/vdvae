from copy import copy
from glob import glob
from time import sleep

import numpy as np
import os
import torch
from tqdm import tqdm
import wandb

import imageio

from vdvae.constants import BASE_DIR
from vdvae.data.data import set_up_data
from vdvae.latents import get_available_latents
from vdvae.train_helpers import set_up_hyperparams, load_vaes
from visualization_scripts.visualize_interpolate import resize
from vdvae.wandb_utils import WANDB_USER, WANDB_DIR


def init_wandb(H):
    tags = ["sample", f"T{H.temp}|{H.temp_rest}"]

    if H.fixed:
        tags.append("fixed")

    wandb.init(project='vae_visualizations', entity=WANDB_USER, dir=WANDB_DIR, tags=tags)
    wandb.config.update({"script": "vis_sample"})

    if H.run_name:
        print(wandb.run.name)
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
    else:
        print(wandb.run.name)
        # wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
        wandb.run.name =  'SAMPLE-' + wandb.run.name

def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default=f'{BASE_DIR}/vdvae/latents/ffhq/all')
    parser.add_argument('--n_files', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--fixed', action="store_true")
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--temp_rest', type=float, default=0)

    # parser.add_argument('--size', type=int, default=128)
    # parser.add_argument('--n_steps', type=int, default=7)
    return parser

def sample_layer(H, l, repr, vae):
    if H.fixed:
        # replace single layer with None such that the model samples it from cond. prior
        zs = repr["z"][:l] + [None] + repr["z"][l+1:]
    else:
        # fix top l latents to "z" (which is sampled from encoder)
        zs = repr["z"][:l]
    # sample next one from posterior by setting temperature to 1
    # keep all other temperatures to 1
    temps = [0] * l + [H.temp] + [H.temp_rest] * (len(repr["z"]) - l - 1)

    return vae.forward_samples_set_latents(1, zs, t=temps)

def visualize(H, file, vae, latent_ids, ls):
    with torch.no_grad():
        z_dict = np.load(file)
        repr = {}
        repr['qm'] = [torch.tensor(z_dict[f'qm_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['pm'] = [torch.tensor(z_dict[f'pm_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['qv'] = [torch.tensor(z_dict[f'qv_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['pv'] = [torch.tensor(z_dict[f'pv_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        repr['z'] = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]

        imgs = []
        variance_images = []
        variance_images_n = []
        for l in ls:
            batch = []
            torch.random.manual_seed(0)
            for i in range(H.n_samples):
                img = sample_layer(H, l, repr, vae)
                img = resize(img, size=(H.size, H.size))
                batch.append(img)

            imgs.extend(batch)
            variances = np.concatenate(batch, axis=0).std(axis=0)[np.newaxis, :, :, :]
            variance_images.append(variances)

            variances = np.repeat(variances.mean(axis=-1)[:, :, :, np.newaxis], 3, axis=-1)

            # variances -= variances.min()
            variances /= variances.max()
            variance_images_n.append((variances * 255).astype('uint8'))
            imgs.append((variances * 255).astype('uint8'))

        # imgs = [resize(img, size=(H.size, H.size)) for img in imgs]
        # variance_images =  [resize(img, size=(H.size, H.size)) for img in variance_images]
        # variance_images_n = [resize(img, size=(H.size, H.size)) for img in variance_images_n]

        n_sampl = H.n_samples + 1
        im = np.concatenate(imgs, axis=0).reshape((len(ls), n_sampl, H.size, H.size, 3)).transpose(
            [0, 2, 1, 3, 4]).reshape([len(ls) * H.size, H.size * n_sampl, 3])

        var_im = np.concatenate(variance_images, axis=0).reshape((len(ls), 1, H.size, H.size, 3)).transpose(
            [0, 2, 1, 3, 4]).reshape([len(ls) * H.size, H.size, 3])
        var_im_n = np.concatenate(variance_images_n, axis=0).reshape((len(ls), 1, H.size, H.size, 3)).transpose(
            [0, 2, 1, 3, 4]).reshape([len(ls) * H.size, H.size, 3])

        i = file.split('/')[-1].split('.')[0]
        wandb.log({"samples": wandb.Image(im, caption=f"{i}")})
        wandb.log({"variances": wandb.Image(var_im, caption=f"{i}")})
        wandb.log({"variances_normalized": wandb.Image(var_im_n, caption=f"{i}")})

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    # ls = [0,1,2,3,7,20,30,40,43,50,60]
    ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 27, 30, 33, 36, 40, 43, 48, 53, 58, 63]
    # ls = list(range(66))  # all layers

    init_wandb(H)
    wandb.config.update({"lv_points": ls})

    files = list(sorted(glob(os.path.join(H.latents_dir, "*.npz"))))[:H.n_files]
    for i, file in tqdm(enumerate(files)):

        visualize(H, file, ema_vae, latent_ids, ls)

if __name__ == "__main__":
    main()
