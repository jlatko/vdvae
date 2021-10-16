import os
from collections import defaultdict

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

def get_classification_score(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }

def run_classifications(cols, layer_ind):
    # TODO: split? kfold?
    # TODO: no missing
    z, meta = get_latents(latents_dir="/scratch/s193223/vdvae/latents/", layer_ind=layer_ind, splits=[1,2,3], allow_missing=False)
    print(z.shape)
    z = z.reshape(z.shape[0], -1)

    scores = {}
    mask_train = meta.split.isin([1,2])
    X_train = z[mask_train]
    meta_train = meta[mask_train]
    X_test = z[~mask_train]
    meta_test = meta[~mask_train]
    for col in cols:
        y_train = np.array(meta_train[col] == 1)
        y_test = np.array(meta_test[col] == 1)
        score = get_classification_score(X_train, X_test, y_train, y_test)
        score["layer_ind"] = layer_ind
        scores[col] = score
        print(col, score)
    return scores

def main():
    cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby", "Attractive"]
    latent_ids = get_available_latents()
    print(latent_ids)
    scores = defaultdict(list)
    for i in latent_ids:
        score_dict = run_classifications(cols, i)
        for col, score in score_dict.items():
            scores[col].append(score)

    for col in cols:
        df = pd.DataFrame(scores[col])
        df.to_csv(f"outputs/{col}.csv", index=False)


if __name__ == "__main__":
    main()