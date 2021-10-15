import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from latents import get_latents, get_available_latents
# TODO: ?
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def get_classification_score(col, layer_ind):
    print(layer_ind, col)
    # TODO: split?
    # TODO: no missing
    z, meta = get_latents(latents_dir="/scratch/s193223/vdvae/latents/", layer_ind=layer_ind, splits=[1], allow_missing=True)
    print(z.shape)
    y = np.array(meta[col] == 1)
    z = z.reshape(z.shape[0], -1)
    # TODO: split?
    X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.33, stratify=y, random_state=42)

    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)

    neigh.fit(X_train, y_train)

    return neigh.score(X_test, y_test)

def main():
    col = "Young"
    latent_ids = get_available_latents()
    print(latent_ids)
    for i in reversed(latent_ids):
        score = get_classification_score(col, i)
        print(i, score)

if __name__ == "__main__":
    main()