import pytorch_lightning as pl
import torch
from torch.optim import AdamW

from train_helpers import update_ema, restore_params, linear_warmup
from utils import get_cpu_stats_over_ranks
from vae import VAE


class Engine(pl.LightningModule):
    def __init__(self, H, preprocess_fn):
        super(Engine, self).__init__()
        self.H = H
        self.preprocess_fn = preprocess_fn
        # self.save_hyperparameters()  # ??
        self.vae = VAE(H)
        self.ema_vae = VAE(H)
        self.skipped_updates = 0
        if H.restore_path:
            print(f'Restoring vae from {H.restore_path}')
            # restore_params(self.vae, H.restore_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
            state_dict = torch.load(H.restore_path, map_location='cpu')
            map_ddp=True
            if map_ddp:
                new_state_dict = {}
                l = len('module.')
                for k in state_dict:
                    if k.startswith('module.'):
                        new_state_dict[k[l:]] = state_dict[k]
                    else:
                        new_state_dict[k] = state_dict[k]
                state_dict = new_state_dict
            self.vae.load_state_dict(state_dict)

        if H.restore_ema_path:
            print(f'Restoring ema vae from {H.restore_ema_path}')
            state_dict = torch.load(H.restore_ema_path, map_location='cpu')
            map_ddp=True
            if map_ddp:
                new_state_dict = {}
                l = len('module.')
                for k in state_dict:
                    if k.startswith('module.'):
                        new_state_dict[k[l:]] = state_dict[k]
                    else:
                        new_state_dict[k] = state_dict[k]
                state_dict = new_state_dict
            self.ema_vae.load_state_dict(state_dict)
            # restore_params(self.ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
        else:
            self.ema_vae.load_state_dict(self.vae.state_dict())

        self.ema_vae.requires_grad_(False)
        print("loaded")

    def on_epoch_end(self) -> None:
        self.skipped_updates = 0

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        update_ema(self.vae, self.ema_vae, self.H.ema_rate)


    def configure_optimizers(self):
        optimizer = AdamW(self.vae.parameters(), weight_decay=self.H.wd, lr=self.H.lr, betas=(self.H.adam_beta1, self.H.adam_beta2))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(self.H.warmup_iters))
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


    def training_step(self, batch, batch_idx):
        print("training step")

        data_input, target = self.preprocess_fn(batch)
        # t0 = time.time()
        stats = self.vae.forward(data_input, target)
        loss = stats['elbo']
        # stats['elbo'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.H.grad_clip).item()
        distortion_nans = torch.isnan(stats['distortion']).sum()
        rate_nans = torch.isnan(stats['rate']).sum()

        stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
        stats = get_cpu_stats_over_ranks(stats)

        cpu_stats = get_cpu_stats_over_ranks(stats)
        # only update if no rank has a nan and if the grad norm is below a specific threshold
        if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (self.H.skip_threshold == -1 or grad_norm < self.H.skip_threshold):
            ok = True
        else:
            ok = False
            self.skipped_updates += 1

        print("inside")

        self.log(
            "skipped_updates",
            self.skipped_updates,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "grad_norm",
            self.grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        print("logged")
        if ok:
            #
            # self.log(
            #     "grad_norm",
            #     self.grad_norm,
            #     on_step=True,
            #     on_epoch=False,
            #     prog_bar=False,
            # )
            # TODO: log
            return loss
        else:
            return