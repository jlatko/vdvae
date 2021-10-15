from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import set_up_data
from train_helpers import set_up_hyperparams, load_vaes

def save_repr(H, ema_vae, data_valid, preprocess_fn):
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    idx = -1
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler)):
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
                    for k in ["z", "kl", "qm", "qv", "pm", "pv"]:
                        stat = block_stats[k][i].cpu().numpy()
                        stat_dict[f"{k}_{block_idx}"] = stat

                np.savez(os.path.join(H.destination_dir, f"{idx}.npz"), **stat_dict)

def add_params(parser):
    parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/latents/')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
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
    save_repr(H, ema_vae, dataset, preprocess_fn)


if __name__ == "__main__":
    main()
