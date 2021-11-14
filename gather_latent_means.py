from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import set_up_data
from train_helpers import set_up_hyperparams, load_vaes
from vae_helpers import gaussian_analytical_kl
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

def update_running_mean(current_mean, new_value, n):
    return current_mean + (new_value - current_mean) / (n + 1)

def get_other_stats(block_stats, qm, pm, qstd, pstd, qv, pv, i):
    other_stats = {}
    other_stats["kl"] = block_stats["kl"][i].cpu().numpy()
    return other_stats

def update_latent_means(stat_dict, block_stats, i, block_idx, n):
    qm = block_stats["qm"][i].cpu().numpy()
    pm = block_stats["pm"][i].cpu().numpy()
    qstd = torch.exp(block_stats["qv"][i]).cpu().numpy()
    pstd = torch.exp(block_stats["pv"][i]).cpu().numpy()
    qv = np.power(qstd, 2)
    pv = np.power(pstd, 2)
    other_stats = get_other_stats(block_stats, qm, pm, qstd, pstd, qv, pv, i)
    if n == 0:
        stat_dict[f"qv_{block_idx}"] = qv
        stat_dict[f"pv_{block_idx}"] = pv
        stat_dict[f"qstd_{block_idx}"] = qstd
        stat_dict[f"pstd_{block_idx}"] = pstd
        stat_dict[f"qm_{block_idx}"] = qm
        stat_dict[f"pm_{block_idx}"] = pm
        for k, v in other_stats:
            stat_dict[f"{k}_{block_idx}"] = v
    else:
        stat_dict[f"qv_{block_idx}"] = update_running_mean(stat_dict[f"qv_{block_idx}"], qv, n)
        stat_dict[f"pv_{block_idx}"] = update_running_mean(stat_dict[f"pv_{block_idx}"], pv, n)
        stat_dict[f"qstd_{block_idx}"] = update_running_mean(stat_dict[f"qstd_{block_idx}"], qstd, n)
        stat_dict[f"pstd_{block_idx}"] = update_running_mean(stat_dict[f"pstd_{block_idx}"], pstd, n)
        stat_dict[f"qm_{block_idx}"] = update_running_mean(stat_dict[f"qm_{block_idx}"], qm, n)
        stat_dict[f"pm_{block_idx}"] = update_running_mean(stat_dict[f"pm_{block_idx}"], pm, n)
        for k, v in other_stats:
            stat_dict[f"{k}_{block_idx}"] = update_running_mean(stat_dict[f"{k}_{block_idx}"], v, n)



def get_stats(H, ema_vae, data_valid, preprocess_fn):
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    idx = -1
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
                if H.dataset == "celebahq":
                    idx = x[1]["idx"][i].item()
                else:
                    idx += 1
                for block_idx, block_stats in enumerate(stats):
                    update_latent_means(stat_dict, block_stats, i, block_idx, n)
                n += 1
        if H.n is not None and n >= H.n:
            break
    np.savez(os.path.join(H.destination_dir, f"{H.dataset}_{H.file_name}.npz"), **stat_dict)
    # all_stats = pd.DataFrame(all_stats)
    # all_stats.to_pickle(os.path.join(H.destination_dir, f"{H.dataset}_latent_stats.pkl"))

def add_params(parser):
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/latent_stats/')
    parser.add_argument('--file_name', type=str, default='latent_means')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
    parser.add_argument('-n', type=int, default=None)

    return parser

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
    vae, ema_vae = load_vaes(H, logprint)
    if H.use_train:
        dataset = data_train
    else:
        dataset = data_valid_or_test

    get_stats(H, ema_vae, dataset, preprocess_fn)

if __name__ == "__main__":
    main()
