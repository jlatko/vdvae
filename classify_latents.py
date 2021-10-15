import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from latents import get_latents, get_available_latents
# TODO: ?
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def get_classification_score(col, layer_ind):
    print(layer_ind, col)
    # TODO: split?
    # TODO: no missing
    z, meta = get_latents(latents_dir="/scratch/s193223/vdvae/latents/", layer_ind=layer_ind, splits=[1,2,3], allow_missing=False)
    print(z.shape)
    y = np.array(meta[col] == 1)
    z = z.reshape(z.shape[0], -1)

    mask_train = meta.split.isin(1,2)
    X_train = z[mask_train]
    y_train = y[mask_train]
    X_test = z[~mask_train]
    y_test = y[~mask_train]

    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)

    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'layer_ind': layer_ind,
    }

def main():
    col = "Young"
    latent_ids = get_available_latents()
    print(latent_ids)
    scores = []
    for i in latent_ids:
        score = get_classification_score(col, i)
        print(score)
        scores.append(score)
    df = pd.DataFrame(scores)
    df.to_csv(f"outputs/{col}.csv", index=False)


if __name__ == "__main__":
    main()