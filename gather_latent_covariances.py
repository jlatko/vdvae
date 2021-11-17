import itertools
from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import set_up_data
from train_helpers import set_up_hyperparams, load_vaes
import pandas as pd

def all_finite(stats):
    for block_idx, block_stats in enumerate(stats):
        qm = block_stats["qm"]
        pm = block_stats["pm"]
        qv = block_stats["qv"]
        pv = torch.exp(block_stats["pv"])
        qstd = torch.exp(block_stats["qv"])
        pstd = torch.exp(block_stats["pv"])
        for x in [qm, pm, qv, pv, qstd, pstd]:
            if not torch.all(torch.isfinite(x)):
                return False
    return True

def update_running_covariance(current_mean, new_value, n):
    return current_mean + (new_value - current_mean) / (n + 1)

def get_current_stats(stats, i, cutoff_masks=None):
    current_stats = {}
    for block_idx, block_stats in enumerate(stats):
        current_stats[f"qm_{block_idx}"] = block_stats["qm"][i].cpu().numpy().reshape(-1)
        current_stats[f"pm_{block_idx}"] = block_stats["pm"][i].cpu().numpy().reshape(-1)
        current_stats[f"qstd_{block_idx}"] = torch.exp(block_stats["qv"][i]).cpu().numpy().reshape(-1)
        current_stats[f"pstd_{block_idx}"] = torch.exp(block_stats["pv"][i]).cpu().numpy().reshape(-1)
        current_stats[f"qv_{block_idx}"] = np.power(current_stats[f"qstd_{block_idx}"], 2).reshape(-1)
        current_stats[f"pv_{block_idx}"] = np.power(current_stats[f"pstd_{block_idx}"], 2).reshape(-1)
        if cutoff_masks is not None:
            current_stats[f"qm_{block_idx}"] = current_stats[f"qm_{block_idx}"][cutoff_masks[f"mask_{i}"]]
            current_stats[f"pm_{block_idx}"] = current_stats[f"pm_{block_idx}"][cutoff_masks[f"mask_{i}"]]
            current_stats[f"qstd_{block_idx}"] = current_stats[f"qstd_{block_idx}"][cutoff_masks[f"mask_{i}"]]
            current_stats[f"pstd_{block_idx}"] = current_stats[f"pstd_{block_idx}"][cutoff_masks[f"mask_{i}"]]
            current_stats[f"qv_{block_idx}"] = current_stats[f"qv_{block_idx}"][cutoff_masks[f"mask_{i}"]]
            current_stats[f"pv_{block_idx}"] = current_stats[f"pv_{block_idx}"][cutoff_masks[f"mask_{i}"]]

    return current_stats


def update_latent_cov(means_dict, stat_dict, current_stats, n, block_pairs, keys):
    deviations = {}
    layers = set(i for i, j in block_pairs) | set(j for i, j in block_pairs)
    for l in layers:
        for k in keys:
            deviations[f"{k}_{l}"] = current_stats[f"{k}_{l}"] - means_dict[f"{k}_{l}"]

    for i, j in block_pairs:
        for k in keys:
            x = np.outer(deviations[f"{k}_{i}"], deviations[f"{k}_{j}"])
            if n == 0:
                stat_dict[f"{k}_{i}_{j}"] = x
            else:
                stat_dict[f"{k}_{i}_{j}"] = update_running_covariance(stat_dict[f"{k}_{i}_{j}"], x, n)

def get_kl_cutoff_mask(means_dict, cutoff):
    cutoff_masks = {}
    for k in means_dict:
        if "kl" in k:
            i = k.split("_")[-1]
            cutoff_masks[f"mask_{i}"] = means_dict[k].reshape(-1) > cutoff
    return cutoff_masks


def update_means_dict(means_dict, cutoff_masks):
    new_dict = {}
    for k in means_dict:
        if "kl" in k:
            i = k.split("_")[-1]
            new_dict[k] = means_dict[k][cutoff_masks[f"mask_{i}"]]
    return new_dict

def get_stats(H, ema_vae, data_valid, preprocess_fn):
    means_dict = {}
    with open(os.path.join(H.means_dir,  f"{H.dataset}_latent_means.npz"), 'rb') as fh:
        npz = np.load(fh)
        for k in npz.keys():
            means_dict[k] = npz[k].reshape(-1)

    cutoff_masks = None
    if H.kl_cutoff is not None:
        cutoff_masks = get_kl_cutoff_mask(means_dict, H.kl_cutoff)
        means_dict = update_means_dict(means_dict, cutoff_masks)

    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    stat_dict = {}
    n = 0
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler)):
        data_input, target = preprocess_fn(x)
        with torch.no_grad():
            stats = ema_vae.forward_get_latents(data_input, get_mean_var=True)
            if not all_finite(stats):
                print("encountered nan/inf, skipping")
                continue
            for i in range(data_input.shape[0]):
                current_stats = get_current_stats(stats, i, cutoff_masks)

                # t = 5
                # block_pairs = \
                #     list(itertools.combinations(range(t), 2)) \
                #       + [(i, i) for i in range(t)] \
                #       + [(i, 20) for i in range(t)] \
                #       + [(i, 43) for i in range(t)] \
                #       + [(43, 43), (20, 20), (20, 43)]
                layers = [1,2, 3, 4, 20, 24, 40]
                block_pairs = \
                    list(itertools.combinations(layers, 2)) \
                      + [(i, i) for i in layers]
                # keys = ["qv", "pv"]
                keys = ["qm", "pm", "qstd", "pstd", "qv", "pv"]

                update_latent_cov(means_dict, stat_dict, current_stats, n, block_pairs, keys)
                n += 1
        if H.n is not None and n >= H.n:
            break
    if cutoff_masks is not None:
        stat_dict.update(cutoff_masks)
    np.savez(os.path.join(H.destination_dir, f"{H.dataset}_{H.file_name}.npz"), **stat_dict)
    # all_stats = pd.DataFrame(all_stats)
    # all_stats.to_pickle(os.path.join(H.destination_dir, f"{H.dataset}_latent_stats.pkl"))

def add_params(parser):
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/latent_stats/')
    parser.add_argument('--means_dir', type=str, default=None)
    parser.add_argument('--file_name', type=str, default='latent_cov')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('--kl_cutoff', type=float, default=None)

    return parser

def main():
    H, logprint = set_up_hyperparams(extra_args_fn=add_params)
    if os.path.exists(H.destination_dir):
        if len(os.listdir(H.destination_dir)) > 0:
            print("WARNING: destination non-empty")
            sleep(5)
            print("continuing")
    else:
        os.makedirs(H.destination_dir)

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.use_train:
        dataset = data_train
    else:
        dataset = data_valid_or_test

    if H.means_dir is None:
        H.means_dir = H.destination_dir

    get_stats(H, ema_vae, dataset, preprocess_fn)

if __name__ == "__main__":
    main()
