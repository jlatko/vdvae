import argparse
import os
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

from hps import Hyperparams

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from latents import get_latents, get_available_latents
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging


def get_classification_score(H, X_train, X_test, y_train, y_test):
    # TODO: normalize??
    if "knn" in H.model:
        n = int(H.model.split("_")[1])
        model = KNeighborsClassifier(n_neighbors=n, n_jobs=H.n_jobs)
    elif H.model == "svc":
        model = SVC(n_jobs=H.n_jobs)
    elif H.model == "logistic":
        model = LogisticRegression(n_jobs=H.n_jobs)
    elif H.model == "l1":
        model = LogisticRegression(n_jobs=H.n_jobs, penalty="l1", C=0.999)
    elif H.model == "rf":
        model = RandomForestClassifier(n_jobs=H.n_jobs)

    else:
        raise ValueError("unknown model")


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc_score': roc_auc_score(y_test, y_pred),
    }

def run_classifications(H, cols, layer_ind):
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
            score = get_classification_score(H, X_train, X_test, y_train, y_test)
            kfold_scores.append(score)
        kfold_scores = pd.DataFrame(kfold_scores)
        score = {}
        for metric in kfold_scores.columns:
            score[f"{metric}_avg"] = kfold_scores[metric].mean()
            score[f"{metric}_std"] = kfold_scores[metric].std()
        score["shape"] = z.shape
        score["layer_ind"] = layer_ind
        score["frequency"] = y.sum() / len(y)
        scores[col] = score
        logging.debug(f"{col}: {score}")
    return scores

def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()


    parser.add_argument('--keys_set', type=str, default='small')
    parser.add_argument('--layer_ids_set', type=str, default='small')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--model', type=str, default='knn_11')
    parser.add_argument('--n_jobs', type=int, default=8)

    H.update(parser.parse_args(s).__dict__)
    return H

def setup(H):
    if H.keys_set == "small":
        cols = ["Young", "Male",  "Smiling", "Wearing_Earrings", "Brown_Hair"]
    elif H.keys_set == "big":
        cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby",
                "Attractive", "Brown_Hair", "Blond_Hair", "Bushy_Eyebrows", "Blurry",
                "Wearing_Earrings", "Heavy_Makeup", "Mouth_Slightly_Open",
                "Narrow_Eyes", "Big_Lips", "Big_Nose", "Eyeglasses", "Wearing_Lipstick"]
    else:
        raise ValueError(f"Unknown keys set {H.keys_set}")

    if H.layer_ids_set == "small":
        latent_ids = [0,1,2,3,5,10,15,20,30,40,50,60]
    elif H.layer_ids_set == "mid":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53, 58, 63]
    elif H.layer_ids_set == "full":
        latent_ids = get_available_latents()
    else:
        raise ValueError(f"Unknown latent ids set {H.layer_ids_set}")

    logging.basicConfig(level=H.log_level)

    return cols, latent_ids

def main():
    H = parse_args()
    cols, latent_ids = setup(H)

    path = f"outputs/{H.model}_{H.run_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    logging.info(cols)
    logging.info(latent_ids)

    scores = defaultdict(list)
    for i in tqdm(latent_ids):
        score_dict = run_classifications(H, cols, i)
        for col, score in score_dict.items():
            scores[col].append(score)

    for col in cols:
        df = pd.DataFrame(scores[col])
        df.to_csv(os.path.join(path, f"{col}.csv"), index=False)


if __name__ == "__main__":
    main()