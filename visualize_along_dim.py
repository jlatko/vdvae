from copy import copy
from time import sleep

import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import imageio

from attributes import get_attributes
from data import set_up_data
from latents import get_available_latents
from train_helpers import set_up_hyperparams, load_vaes
import wandb

import torch.nn.functional as F

from visualize_along_attr import get_zs_for_idx


def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--latent_dim_file', type=str, default=None)
    parser.add_argument('--norm', type=str, default="pixel", help="none|channel|pixel")
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_dim', type=int, default=8)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=13)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--latent_key', type=str, default="z")

    return parser

def resize(img, size):
    img = Image.fromarray(img.squeeze())
    img = img.resize(size=size)
    img = np.array(img)[np.newaxis]
    return img

def scale_direction(direction, normalize=None, scale=1):
    if normalize == "channel":
        dim = [1]
        norm = torch.norm(direction, p=2, dim=dim, keepdim=True)
        direction = direction.div(norm.expand_as(direction))
        scale = np.sqrt(scale * direction.shape[1])
    elif normalize == "pixel":
        dim = [1,2,3]
        norm = torch.norm(direction, p=2, dim=dim, keepdim=True)
        direction = direction.div(norm.expand_as(direction))
        scale = np.sqrt(scale * direction.shape[1] * direction.shape[2] * direction.shape[3])

    return scale * direction

def get_direction(H, dims, k, l, dim_i):
    # get direction
    mask = dims[f"mask_{l}"]
    s = len(mask)
    direction = np.zeros(s)

    x = dims[f"{k}_d_{l}"][:,dim_i]

    res = np.sqrt(s /16)
    shape = (16, int(res), int(res))
    print(shape)

    direction[mask] = x
    direction = direction.reshape(shape)

    wandb.log({f"std_{l}_{dim_i}": direction.std(), "i": dim_i})
    direction = torch.tensor(direction[np.newaxis], dtype=torch.float32).cuda()
    direction = scale_direction(direction, normalize=H.norm, scale=H.scale)
    wandb.log({f"scaled_std_{l}_{dim_i}": torch.std(direction).item(), "i": dim_i})

    return direction


def dim_manipulation(H, k, ema_vae, latent_ids, metadata, dims, i=0):
    with torch.no_grad():

        idx = metadata.iloc[i].idx
        zs = get_zs_for_idx(H, idx, latent_ids)

        batches = []
        for dim_i in range(H.n_dim):

            layers = [dims["l1"], dims["l2"]]
            for l_ind in layers:
                torch.random.manual_seed(0)

                zs_current = copy(zs)

                direction = get_direction(H, dims, k, l_ind, dim_i)

                for a in np.linspace(-1, 1, H.n_steps):
                    zs_current[l_ind] = zs[l_ind] + a * direction
                    if H.fixed:
                        img = ema_vae.forward_samples_set_latents(1, zs_current, t=H.temp)
                    else:
                        img = ema_vae.forward_samples_set_latents(1, zs_current[:l_ind+1], t=H.temp)

                    img = resize(img, size=(H.size, H.size))
                    batches.append(img)

        n_rows = 2 * H.n_dim
        im = np.concatenate(batches, axis=0).reshape((n_rows,  H.n_steps, H.size, H.size, 3)).transpose(
            [0, 2, 1, 3, 4]).reshape([n_rows * H.size, H.size * H.n_steps, 3])

        # add lines
        for j in range(1, H.n_dim):
            im[j*2*H.size - 2 : j*2*H.size + 2] = 0
            im[j*2*H.size - 2 : j*2*H.size + 2,:,0] = 150

        name_key = f"{k}_{layers[0]}x{layers[1]}_t{str(H.temp).replace('.','_')}_"
        if H.fixed:
            name_key += "fixed_"


        fname = os.path.join(wandb.run.dir, f"{name_key}{idx}.png")
        imageio.imwrite(fname, im)
        wandb.log({f"{name_key}": wandb.Image(im, caption=f"{name_key}_{idx}")})


def init_wandb(H, dims):
    tags = []
    tags.append(dims["method"].item())
    tags.append(f'{dims["l1"]}_{dims["l2"]}')
    if H.fixed:
        tags.append("fixed")

    wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/", tags=tags)
    wandb.config.update({"script": "vis_dim"})

    print(wandb.run.name)
    if H.run_name:
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
    else:
        run_name = H.latent_dim_file.split("/")[-1].split(".")[0]
        run_name += "_" + str(H.temp)

        if H.fixed:
            run_name += "_fixed"

        wandb.run.name = run_name + "_" + H.latent_key +  '-' + wandb.run.name.split('-')[-1]

def plot_metrics(dims):
    l = len(dims[f"{dims['keys'][0]}_explained_var"])
    for i in range(l):
        for k in dims["keys"]:
            wandb.log({
                f"{k}_explained_var": dims[f"{k}_explained_var"],
                # f"{k}_explained_var1": dims[f"{k}_explained_var1"],
                # f"{k}_explained_var2": dims[f"{k}_explained_var2"],
                # f"{k}_covariance_12": dims[f"{k}_covariance_12"],
                f"{k}_correlation_12": dims[f"{k}_correlation_12"],
            }, step=i)

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)
    assert H.latent_dim_file is not None

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)

    dims = np.load(H.latent_dim_file)

    init_wandb(H, dims)
    wandb.config.update(H)

    plot_metrics(dims)

    latent_ids = get_available_latents(H.latents_dir)
    for i in range(H.n_samples):
        for k in dims["keys"]:
            dim_manipulation(H, k=k, ema_vae=ema_vae, latent_ids=latent_ids, dims=dims, i=i, metadata=data_valid_or_test.metadata)

if __name__ == "__main__":
    main()
