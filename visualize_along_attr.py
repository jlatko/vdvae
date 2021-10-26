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

from data import set_up_data
from latents import get_available_latents
from train_helpers import set_up_hyperparams, load_vaes
import wandb

import torch.nn.functional as F

wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/")
wandb.config.update({"script": "vis_attr"})


def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--attr_means_dir', type=str, default='/scratch/s193223/vdvae/attr_means/')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=15)
    return parser

def resize(img, size):
    img = Image.fromarray(img.squeeze())
    img = img.resize(size=size)
    img = np.array(img)[np.newaxis]
    return img

def attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, lv_points, fixed=True, temp=0.1, normalize=True):
    with torch.no_grad():
        z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
        zs = [torch.tensor(z_dict[f'z_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]
        height = 64
        width = 64

        for attr in tqdm(attributes):
            batches = []
            for i in lv_points:
                zs_current = copy(zs)
                # get direction
                means_dict = np.load(os.path.join(H.attr_means_dir, f"{i}.npz"))
                direction = means_dict[f"{attr}_neg"] - means_dict[f"{attr}_pos"]
                wandb.log({f"std_{attr}_{idx}": direction.std(), "i": i})
                direction = torch.tensor(direction[np.newaxis], dtype=torch.float32).cuda()
                # norm
                dim = [1] # TODO: consider different norm [1]? [2,3]?
                norm = torch.norm(direction, p=2, dim=dim, keepdim=True)
                direction = direction.div(norm.expand_as(direction))
                # print(direction)
                # direction = direction.div(direction.std())
                # direction = F.normalize(direction, p=2)

                if normalize:
                    scale = 2
                else:
                    scale = 10


                for a in np.linspace(-scale, scale, H.n_steps):
                    zs_current[i] = zs[i] + a * direction
                    if fixed:
                        img = ema_vae.forward_samples_set_latents(1, zs_current, t=temp)
                    else:
                        img = ema_vae.forward_samples_set_latents(1, zs_current[:i+1], t=temp)

                    img = resize(img, size=(height, width))
                    batches.append(img)
            n_rows = len(lv_points)
            #TODO: consider downsampling
            im = np.concatenate(batches, axis=0).reshape((n_rows,  H.n_steps, height, width)).transpose(
                [0, 2, 1, 3, 4]).reshape([n_rows * height, width * H.n_steps, 3])

            name_key = f"t{str(temp).replace('.','_')}_"
            if fixed:
                name_key += "fixed_"

            fname = os.path.join(wandb.run.dir, f"{attr}_{name_key}{idx}.png")
            imageio.imwrite(fname, im)

            wandb.log({f"{attr}_{name_key}": wandb.Image(im, caption=f"{attr}_{name_key}{idx}")})

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)


    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    if H.run_name:
        print(wandb.run.name)
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]

    attributes = ["Young", "Male", "Smiling", "Wearing_Earrings", "Brown_Hair", "Blond_Hair", "Attractive"]
    lv_points = [0,1,2,3,4,5,6,7,20,21,40, 41, 43, 51, 60]

    wandb.config.update({"attributes": attributes, "latent_ids": latent_ids, "lv_points": lv_points})
    print(lv_points)
    for i in range(H.n_samples):
        idx = data_valid_or_test.metadata.iloc[i].idx
        attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, lv_points, fixed=False)
        attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, lv_points, fixed=False, temp=0.2)
        attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, lv_points, fixed=False, temp=0.5)
        # attribute_manipulation(H, idx, attributes, ema_vae, latent_ids, lv_points, fixed=True)


if __name__ == "__main__":
    main()
