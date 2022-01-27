from time import sleep

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vdvae.data.data import set_up_data
from vdvae.train_helpers import set_up_hyperparams, load_vaes, parse_hparams, setup_parsed
from vdvae.model.vae_helpers import gaussian_analytical_kl
import pandas as pd
import wandb


def setup_wandb(H):
    # add tags and initialize wandb run
    tags = ["stats"]

    wandb.init(project="vdvae_analysis", entity="johnnysummer", dir='/scratch/s193223/wandb', tags=tags)
    wandb.config.update(H)

    # wandb configuration
    if H.run_name is not None:
        run_name = H.run_name
    else:
        run_name = H.dataset

    run_name = "STATS_" + run_name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.run.save()

def get_kls(block_stats, i, block_idx):
    qm = block_stats["qm"][i]
    pm = block_stats["pm"][i]
    qv = block_stats["qv"][i]
    pv = block_stats["pv"][i]
    kl_forward = gaussian_analytical_kl(pm, qm, pv, qv)
    return {
        f"kl_forward_mean_{block_idx}": torch.mean(kl_forward).cpu().numpy().astype(np.float16),
        f"kl_forward_std_{block_idx}": torch.std(kl_forward).cpu().numpy().astype(np.float16),
        f"kl_mean_{block_idx}": torch.mean(block_stats["kl"][i]).cpu().numpy().astype(np.float16),
        f"kl_std_{block_idx}": torch.std(block_stats["kl"][i]).cpu().numpy().astype(np.float16)
    }

def get_basic_stats(block_stats, i, block_idx):
    qm = block_stats["qm"][i]
    pm = block_stats["pm"][i]
    qstd = torch.exp(block_stats["qv"][i])
    pstd = torch.exp(block_stats["pv"][i])
    qv = torch.pow(qstd, 2)
    pv = torch.pow(pstd, 2)
    return {
        f"qv_mean_{block_idx}": torch.mean(qv).cpu().numpy().item(),
        f"qstd_mean_{block_idx}": torch.mean(qstd).cpu().numpy().item(),
        f"qv_mean_squared_{block_idx}":torch.mean(torch.pow(qv, 2)).cpu().numpy().item(),
        f"pv_mean_{block_idx}": torch.mean(pv).cpu().numpy().item(),
        f"pstd_mean_{block_idx}": torch.mean(pstd).cpu().numpy().item(),
        f"pv_mean_squared_{block_idx}":torch.mean(torch.pow(pv, 2)).cpu().numpy().item(),
        f"pm_mean_squared_{block_idx}":torch.mean(torch.pow(pm, 2)).cpu().numpy().item(),
        f"qm_mean_squared_{block_idx}": torch.mean(torch.pow(qm, 2)).cpu().numpy().item(),
        f"pm_mean_abs_{block_idx}":torch.mean(torch.abs(pm)).cpu().numpy().item(),
        f"qm_mean_abs_{block_idx}": torch.mean(torch.abs(qm)).cpu().numpy().item(),
        f"mean_diff_sq_{block_idx}": np.power(qm.cpu().numpy() - pm.cpu().numpy(), 2).mean().item(),
        f"mean_diff_abs_{block_idx}": np.abs(qm.cpu().numpy() - pm.cpu().numpy()).mean().item(),
        f"var_diff_sq_{block_idx}": np.power(qv.cpu().numpy() - pv.cpu().numpy(), 2).mean().item(),
        f"var_diff_abs_{block_idx}": np.abs(qv.cpu().numpy() - pv.cpu().numpy()).mean().item(),
        f"std_diff_abs_{block_idx}": np.abs(qstd.cpu().numpy() - pstd.cpu().numpy()).mean().item(),
    }


def get_stats(H, ema_vae, data_valid, preprocess_fn):
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    idx = -1
    all_stats = []
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler)):
        data_input, target = preprocess_fn(x)
        with torch.no_grad():
            stats = ema_vae.forward_get_latents(data_input, get_mean_var=True)
            for i in range(data_input.shape[0]):
                stat_dict = {}

                if H.dataset == "celebahq":
                    idx = x[1]["idx"][i].item()
                else:
                    idx += 1

                stat_dict["idx"] = idx
                for block_idx, block_stats in enumerate(stats):
                    # for k in keys:
                    #     stat = block_stats[k][i].cpu().numpy().astype(np.float16)
                    #     if not np.isfinite(stat).all():
                    #         print(f"WARNING: {idx}: {k}_{block_idx} contains NaN or inf")

                    stat_dict.update(get_kls(block_stats, i, block_idx))
                    stat_dict.update(get_basic_stats(block_stats, i, block_idx))

                # np.savez(os.path.join(H.destination_dir, f"{idx}.npz"), **stat_dict)
                # TODO: save everything to a single file
                all_stats.append(stat_dict)
        if H.n is not None and len(all_stats) >= H.n:
            break
    all_stats = pd.DataFrame(all_stats)
    all_stats.to_pickle(os.path.join(H.destination_dir, f"{H.dataset}_{H.file_name}.pkl"))

def add_params(parser):
    # parser.add_argument('--destination_dir', type=str, default='/scratch/s193223/vdvae/latent_stats/')
    parser.add_argument('--use_train', dest='use_train', action='store_true')
    parser.add_argument('--file_name', type=str, default='latent_stats')
    parser.add_argument('-n', type=int, default=None)

    return parser

def main():
    H = parse_hparams(extra_args_fn=add_params)
    setup_wandb(H)

    if os.path.exists(H.destination_dir):
        if len(os.listdir(H.destination_dir)) > 0:
            print("WARNING: destination non-empty")
            sleep(5)
            print("continuing")
        #     raise RuntimeError('Destination non empty')
    else:
        os.makedirs(H.destination_dir)

    H.destination_dir = wandb.run.dir
    logprint = setup_parsed(H, dir=os.path.join(wandb.run.dir, 'log'))

    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.use_train:
        dataset = data_train
    else:
        dataset = data_valid_or_test

    get_stats(H, ema_vae, dataset, preprocess_fn)

if __name__ == "__main__":
    main()
