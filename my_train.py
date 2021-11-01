import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import pytorch_lightning as pl

from data import set_up_data
from engine import Engine
from utils import get_cpu_stats_over_ranks
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema, \
    set_up_hyperparams_light

import wandb
wandb.init(project='vdvae', entity='johnnysummer', dir="/scratch/s193223/wandb/")
wandb.config.update({"script": "my_train"})



def main():
    H, logprint = set_up_hyperparams_light(dir=os.path.join(wandb.run.dir, 'log'))
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    engine = Engine(H, preprocess_fn)

    dataloader_train = DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, shuffle=True)
    dataloader_val = DataLoader(data_valid_or_test, batch_size=H.n_batch, drop_last=True, pin_memory=True, shuffle=False)

    if H.run_name is not None:
        wandb.run.name = H.run_name
        wandb.run.save()
    H.save_dir = wandb.run.dir # ???
    wandb.config.update(H)
    wandb.config.update({"machine": os.uname()[1]})
    wandb.save('*.png')
    wandb.save('*.th')

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(patience=10, monitor="loss"))
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=wandb.run.dir,
            monitor="loss",
            filename="model",
            verbose=True,
            period=1,
        )
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus=gpus,
        precision=16,
        accumulate_grad_batches=32,
        # limit_train_batches=10,
        # limit_test_batches=1,
        # **cfg["trainer"],
    )
    trainer.fit(engine, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)

if __name__ == "__main__":
    main()
