from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vdvae.data.data import set_up_data
from vdvae.train_helpers import set_up_hyperparams, load_vaes

def save_repr(H, ema_vae, data_valid, preprocess_fn, keys=("z", "kl", "qm", "qv", "pm", "pv")):
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    idx = -1
    n = 0
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler)):
        if H.check_files:
            all_present  = True
            for i in range(x.shape[0]):
                if H.dataset == "celebahq":
                    idx = x[1]["idx"][i].item()
                else:
                    idx += 1
                if not os.os.path.join(H.destination_dir, f"{idx}.npz")
            if all_present:
                continue

        data_input, target = preprocess_fn(x)
        with torch.no_grad():
            stats = ema_vae.forward_get_latents(data_input, get_mean_var=True)
            # stats = get_cpu_stats_over_ranks(stats)
            for i in range(data_input.shape[0]):
                stat_dict = {}
                if H.dataset == "celebahq":
                    idx = x[1]["idx"][i].item()
                else:
                    idx += 1
                for block_idx, block_stats in enumerate(stats):
                    for k in keys:
                        stat = block_stats[k][i].cpu().numpy().astype(np.float16)
                        if not np.isfinite(stat).all():
                            print(f"WARNING: {idx}: {k}_{block_idx} contains NaN or inf")
                        stat_dict[f"{k}_{block_idx}"] = stat

                np.savez(os.path.join(H.destination_dir, f"{idx}.npz"), **stat_dict)
                n += 1
                if H.n is not None and n >= H.n:
                    return

def add_params(parser):
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
    parser.add_argument('--check_files', dest='check_files', action='store_true')
    parser.add_argument('--keys_mode', type=str, default='z')
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

    keys = H.keys_mode.split(',')
    assert len(set(keys) - {"z", "kl", "qm", "qv", "pm", "pv"}) == 0
    save_repr(H, ema_vae, dataset, preprocess_fn, keys=keys)


if __name__ == "__main__":
    main()
