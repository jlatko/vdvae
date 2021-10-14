from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema


def main():
    H, logprint = set_up_hyperparams()
    H.image_channels = 3
    H.image_size = 256
    # H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    import IPython
    IPython.embed()
    import numpy as np
    import torch
    img = np.random.random(size=(1, 256, 256, 3))
    b = torch.tensor(img, dtype=torch.float)
    latents = ema_vae.forward_get_latents(b)
    zs = [s['z'] for s in ema_vae.forward_get_latents(b)]


if __name__ == "__main__":
    main()
