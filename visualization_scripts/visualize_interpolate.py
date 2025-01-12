from copy import copy

import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
import imageio

from vdvae.constants import BASE_DIR
from vdvae.attributes import get_attributes
from vdvae.data.data import set_up_data
from vdvae.latents import get_available_latents
from vdvae.train_helpers import set_up_hyperparams, load_vaes
import wandb
from vdvae.wandb_utils import _download, WANDB_USER, WANDB_DIR

MIN_FREQ = 2

def add_params(parser):
    parser.add_argument('--latents_dir', type=str, default=f'{BASE_DIR}/vdvae/latents/')
    parser.add_argument('--attr_means_dir', type=str, default=f'{BASE_DIR}/vdvae/attr_means/')
    parser.add_argument('--norm', type=str, default="pixel", help="none|channel|pixel")
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=13)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--keys_set', type=str, default='small')
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--fixed', action="store_true")
    parser.add_argument('--grouped', action="store_true")
    parser.add_argument('--has_attr', action="store_true")
    parser.add_argument('--use_group_direction', action="store_true")
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

def get_idx_for_attr(H, attr, has_attr, metadata, i):
    attr_mask = metadata[attr] == 1
    male_mask = metadata["Male"] == 1
    ignore_male = (attr_mask & male_mask).sum() < MIN_FREQ or ((~attr_mask) & male_mask).sum() < MIN_FREQ
    ignore_female = (attr_mask & (~male_mask)).sum() < MIN_FREQ or ((~attr_mask) & (~male_mask)).sum() < MIN_FREQ

    if ignore_male:
        mask = ~male_mask
    elif ignore_female:
        mask = male_mask

    if (ignore_female and ignore_male) or (not ignore_female and not ignore_male):
        mask = metadata[attr].isin([-1,1]) # this should be all true
        assert mask.all()

    if not has_attr:
        attr_mask = ~attr_mask

    return metadata[mask & attr_mask].iloc[i].idx

def get_zs_for_idx(H, idx, latent_ids):
    z_dict = np.load(os.path.join(H.latents_dir, f"{idx}.npz"))
    return [torch.tensor(z_dict[f'{H.latent_key}_{i}'][np.newaxis], dtype=torch.float32).cuda() for i in latent_ids]

def get_direction(H, attr, i, idx, sample_meta):
    # get direction
    means_dict = np.load(os.path.join(H.attr_means_dir, f"{i}.npz"))
    if H.use_group_direction and attr != "Male":
        if sample_meta["Male"]:
            direction = means_dict[f"{attr}_diff_male"]
        else:
            direction = means_dict[f"{attr}_diff_female"]
    elif H.grouped and attr != "Male":
        direction = means_dict[f"{attr}_diff_grouped"]
    else:
        # direction = means_dict[f"{attr}_diff"]
        direction = means_dict[f"{attr}_pos"] - means_dict[f"{attr}_neg"]
    wandb.log({f"std_{attr}_{idx}": direction.std(), "i": i})
    direction = torch.tensor(direction[np.newaxis], dtype=torch.float32).cuda()
    direction = scale_direction(direction, normalize=H.norm, scale=H.scale)
    wandb.log({f"scaled_std_{attr}_{idx}": torch.std(direction).item(), "i": i})

    return direction

def attribute_manipulation(H, attributes, ema_vae, latent_ids, lv_points, metadata, idx=None, has_attr=None, i=0):
    with torch.no_grad():

        if has_attr is None:
            idx = metadata.iloc[i].idx
            zs = get_zs_for_idx(H, idx, latent_ids)
            sample_meta = metadata.set_index("idx").loc[idx]

        for attr in tqdm(attributes):
            try:
                if has_attr is not None:
                    idx = get_idx_for_attr(H, attr, has_attr, metadata, i)
                    zs = get_zs_for_idx(H, idx, latent_ids)
                    sample_meta = metadata.set_index("idx").loc[idx]

                batches = []
                for l_ind in lv_points:
                    torch.random.manual_seed(0)

                    zs_current = copy(zs)

                    direction = get_direction(H, attr, l_ind, idx, sample_meta)

                    for a in np.linspace(-1, 1, H.n_steps):
                        zs_current[l_ind] = zs[l_ind] + a * direction
                        if H.fixed:
                            img = ema_vae.forward_samples_set_latents(1, zs_current, t=H.temp)
                        else:
                            img = ema_vae.forward_samples_set_latents(1, zs_current[:l_ind+1], t=H.temp)

                        img = resize(img, size=(H.size, H.size))
                        batches.append(img)
                n_rows = len(lv_points)
                #TODO: consider downsampling
                im = np.concatenate(batches, axis=0).reshape((n_rows,  H.n_steps, H.size, H.size, 3)).transpose(
                    [0, 2, 1, 3, 4]).reshape([n_rows * H.size, H.size * H.n_steps, 3])

                name_key = f"t{str(H.temp).replace('.','_')}_"
                if H.fixed:
                    name_key += "fixed_"

                fname = os.path.join(wandb.run.dir, f"{attr}_{name_key}{idx}.png")
                imageio.imwrite(fname, im)

                wandb.log({f"{attr}_{name_key}": wandb.Image(im, caption=f"{attr}_{name_key}{idx}")})
            except KeyError as e:
                print(e)
                print("cont")

def init_wandb(H):
    tags = []
    if H.grouped:
        tags.append("grouped")
    if H.use_group_direction:
        tags.append("group_direction")
    if H.has_attr:
        tags.append("has_attr")
    if H.fixed:
        tags.append("fixed")

    wandb.init(project='vae_visualizations', entity=WANDB_USER, dir=WANDB_DIR, tags=tags)
    wandb.config.update({"script": "vis_attr"})

    if H.run_name:
        print(wandb.run.name)
        wandb.run.name = H.run_name + '-' + wandb.run.name.split('-')[-1]
    else:
        print(wandb.run.name)
        run_name = H.keys_set
        # if H.latents_dir != f'{BASE_DIR}/vdvae/latents/':
        #     run_name += "_tuned"
        run_name += "_" + str(H.temp)

        if H.fixed:
            run_name += "_fixed"
        if H.use_group_direction:
            run_name += "_group_d"

        elif H.grouped:
            run_name += "_grouped"

        if H.has_attr:
            run_name += "_has_attr"

        wandb.run.name = run_name + "_" + H.latent_key +  '-' + wandb.run.name.split('-')[-1]

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)


    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)


    latent_ids = get_available_latents(H.latents_dir)
    vae, ema_vae = load_vaes(H, logprint)

    init_wandb(H)


    attributes = get_attributes(H.keys_set)

    # lv_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 27, 30, 33, 36, 40, 43, 48, 53, 58, 63]
    lv_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 27, 30, 33, 36, 40, 43, 48, 53]

    wandb.config.update(H)
    wandb.config.update({"attributes": attributes, "latent_ids": latent_ids, "lv_points": lv_points})

    print(lv_points)
    if H.has_attr:
        for i in range(H.n_samples):
            for has_attr in [True, False]:
                attribute_manipulation(H, attributes, ema_vae, latent_ids, lv_points, has_attr=has_attr, metadata=data_valid_or_test.metadata, i=i)

    else:
        for i in range(H.n_samples):
            attribute_manipulation(H, attributes, ema_vae, latent_ids, lv_points, i=i, metadata=data_valid_or_test.metadata)

if __name__ == "__main__":
    main()
