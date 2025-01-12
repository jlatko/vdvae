from datetime import datetime

import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from tqdm import tqdm

from vdvae.data.data import set_up_data
from vdvae.utils import get_cpu_stats_over_ranks, mpi_size, mpi_rank
from vdvae.train_helpers import load_vaes, load_opt, accumulate_stats, save_model, update_ema, \
    parse_hparams, setup_parsed
from vdvae.wandb_utils import WANDB_USER, WANDB_DIR
from vdvae.constants import BASE_DIR

import wandb


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    vae.zero_grad()
    stats = vae.forward(data_input, target)
    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats = ema_vae.forward(data_input, target)
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset in ('ffhq_1024', 'celebahq') else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)
    train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            data_input, target = preprocess_fn(x)
            with autocast():
                training_stats = training_step(H, data_input, target, vae, ema_vae, optimizer, iterate)
                stats.append(training_stats)
                scheduler.step()

            if iterate % 100 == 0:
                accumulated_stats =  accumulate_stats(stats, 100)
                accumulated_stats['epoch'] = epoch
                accumulated_stats['batch'] = iterate
                accumulated_stats['lr'] = scheduler.get_last_lr()[0]
                wandb.log(accumulated_stats, step=iterate)
                if accumulated_stats["skipped_updates"] == 100:
                    print("Skipped all updates, lowering LR...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                        break

            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed, iterate, logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)
            valid_stats['epoch'] = epoch
            wandb.log(valid_stats, step=iterate)


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    for x in tqdm(DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler)):
        data_input, target = preprocess_fn(x)
        stats_valid.append(eval_step(data_input, target, ema_vae))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_elbo=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, iterate, logprint):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        # create reconstructions using i first latents and low temperature (0.1)
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    logprint(f'printing samples for {iterate}')
    fname = f'{H.save_dir}/samples-{iterate}.png'
    imageio.imwrite(fname, im)

    images = wandb.Image(
        im,
        caption=f"samples-{iterate}",
    )
    wandb.log({"samples": images}, step=iterate)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    wandb.log(stats)
    logprint(type='test_loss', **stats)


def main():
    H = parse_hparams()
    run_name, group_name = None, None
    time_str = datetime.now().strftime("%d_%m__%H_%M")

    if H.run_name is not None:
        run_name = H.run_name
    else:
        run_name = f"{H.dataset}_{time_str}"

    if mpi_size() > 1:
        run_name = f"DDP{mpi_size()}_" + run_name + '-' + str(mpi_rank())
        group_name = f"DDP{mpi_size()}_{H.dataset}_{time_str}"

    if H.test_eval:
        tags = ["eval"]
        if group_name is None:
            group_name = "eval"
    else:
        tags = ["train"]
        if group_name is None:
            group_name = "train"

    if H.cifar_group is not None:
        tags.append(H.cifar_group)

    wandb.init(project='vdvae', entity=WANDB_USER, dir=WANDB_DIR, tags=tags, name=run_name, group=group_name)
    H.save_dir = wandb.run.dir # ???

    logprint = setup_parsed(H, dir=os.path.join(wandb.run.dir, 'log'))
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    vae, ema_vae = load_vaes(H, logprint)

    wandb.config.update(H)
    wandb.config.update({"machine": os.uname()[1]})
    wandb.save(os.path.join(wandb.run.dir, '*.png'))
    wandb.save(os.path.join(wandb.run.dir, '*.th'))

    if H.test_eval:
        # wandb.run.tags.append("eval")
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
    else:
        # wandb.run.tags.append("train")
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint)


if __name__ == "__main__":
    main()
    wandb.finish()
