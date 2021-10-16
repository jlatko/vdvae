import os
from collections import defaultdict

from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from latents import get_latents, get_available_latents
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)

def get_classification_score(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }

def run_classifications(cols, layer_ind):
    z, meta = get_latents(latents_dir="/scratch/s193223/vdvae/latents/", layer_ind=layer_ind, splits=[1,2,3], allow_missing=False)
    logging.debug(z.shape)
    z = z.reshape(z.shape[0], -1)

    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    scores = {}
    for col in cols:
        kfold_scores = []
        y = np.array(meta[col] == 1)
        for train_index, test_index in kfold.split(z, y):
            X_train, X_test = z[train_index], z[test_index]
            y_train, y_test = y[train_index], y[test_index]
            score = get_classification_score(X_train, X_test, y_train, y_test)
            kfold_scores.append(score)
        kfold_scores = pd.DataFrame(kfold_scores)
        score = {}
        for metric in kfold_scores.columns:
            score[f"{metric}_avg"] = kfold_scores[metric].mean()
            score[f"{metric}_std"] = kfold_scores[metric].std()
        score["shape"] = z.shape
        score["layer_ind"] = layer_ind
        scores[col] = score
        logging.debug(f"{col}: {score}")
    return scores

def main():
    cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby",
            "Attractive", "Brown_Hair", "Blond_Hair", "Bushy_Eyebrows", "Blurry",
            "Wearing_Earrings", "Heavy_Makeup", "Mouth_Slightly_Open",
            "Narrow_Eyes", "Big_Lips", "Big_Nose", "Eyeglasses", "Wearing_Lipstick"]

    latent_ids = get_available_latents()
    logging.info(latent_ids)
    scores = defaultdict(list)
    for i in tqdm(latent_ids):
        score_dict = run_classifications(cols, i)
        for col, score in score_dict.items():
            scores[col].append(score)

    for col in cols:
        df = pd.DataFrame(scores[col])
        df.to_csv(f"outputs/{col}.csv", index=False)


if __name__ == "__main__":
    main()