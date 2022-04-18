import os
from vdvae.constants import WANDB_DIR, WANDB_USER
# change those to appropriate w&b username and path



def _download(file, path, force_redownload=False):
    full_path = os.path.join(path, file.name)
    if os.path.exists(full_path) and not force_redownload:
        return full_path
    else:
        file.download(path, replace=True)
        return full_path