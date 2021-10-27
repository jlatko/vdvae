import argparse
import os

from attributes import get_attributes

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from collections import defaultdict

if os.environ["CUDA_VISIBLE_DEVICES"]:
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml import LogisticRegression as cuLogisticRegression
    from cuml.linear_model import MBSGDClassifier as cuMBSGDClassifier
    from cuml.svm import SVC as cuSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from tqdm import tqdm

from hps import Hyperparams

import numpy as np
import pandas as pd
from latents import get_latents, get_available_latents
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

import wandb
wandb.init(project='vdvae_analysis', entity='johnnysummer', dir="/scratch/s193223/wandb/")


# TODO: consider accounting for the most frequent and correlated attributes (earings only for females etc)

def get_classification_score(H, X_train, X_test, y_train, y_test, cuda=False):
    # TODO: normalize??
    if "knn" in H.model:
        n = int(H.model.split("_")[1])
        if cuda:
            model = cuKNeighborsClassifier(n_neighbors=n)
        else:
            model = KNeighborsClassifier(n_neighbors=n, n_jobs=H.n_jobs)
    elif H.model == "svc":
        if cuda:
            model = cuSVC()
        else:
            model = SVC(n_jobs=H.n_jobs)
    elif H.model == "logistic":
        if cuda:
            # model = cuLogisticRegression(n_jobs=H.n_jobs)
            model = cuMBSGDClassifier(loss="log", penalty="none")
        else:
            model = LogisticRegression(n_jobs=H.n_jobs, solver="saga", max_iter=500)
    elif H.model == "l1":
        if cuda:
            # model = cuLogisticRegression(n_jobs=H.n_jobs, penalty="l1", C=0.999)
            model = cuMBSGDClassifier(loss="log", penalty="l1", alpha=0.0001)
        else:
            model = LogisticRegression(n_jobs=H.n_jobs, penalty="l1", C=0.999, solver="saga", max_iter=500)
    elif H.model == "l2":
        if cuda:
            # model = cuLogisticRegression(n_jobs=H.n_jobs, penalty="l2", C=0.999)
            model = cuMBSGDClassifier(loss="log", penalty="l2", alpha=0.0001)
        else:
            model = LogisticRegression(n_jobs=H.n_jobs, penalty="l2", C=0.999, solver="saga", max_iter=500)
    elif H.model == "rf":
        if cuda:
            model = cuRandomForestClassifier()
        else:
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

def run_classifications(H, cols, layer_ind, latents_dir, handle_nan=False, cuda=False):
    z, meta = get_latents(latents_dir=latents_dir, layer_ind=layer_ind, splits=[1,2,3], allow_missing=False, handle_nan=handle_nan)
    logging.debug(z.shape)

    resolution = z.shape[-2]
    wandb.log({"resolution": resolution}, step=layer_ind)
    z = z.reshape(z.shape[0], -1)
    wandb.log({"size": z.shape[1]}, step=layer_ind)
    # TODO: ? move to gpu? X_cudf = cudf.DataFrame(X)

    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    scores = {}
    for col in cols:
        kfold_scores = []
        y = np.array(meta[col] == 1)
        for train_index, test_index in kfold.split(z, y):
            X_train, X_test = z[train_index], z[test_index]
            y_train, y_test = y[train_index], y[test_index]
            score = get_classification_score(H, X_train, X_test, y_train, y_test, cuda=cuda)
            kfold_scores.append(score)
        kfold_scores = pd.DataFrame(kfold_scores)
        score = {}
        for metric in kfold_scores.columns:
            score[f"{metric}_avg"] = kfold_scores[metric].mean()
            score[f"{metric}_std"] = kfold_scores[metric].std()

        wandb.log({
            f"{col}_{k}": v
            for k, v
            in score.items()
        }, step=layer_ind)

        score["shape"] = z.shape
        score["layer_ind"] = layer_ind
        score["frequency"] = y.sum() / len(y)
        scores[col] = score
        logging.debug(f"{col}: {score}")
    return scores

def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--latents_dir', type=str, default="/scratch/s193223/vdvae/latents/")
    parser.add_argument('--keys_set', type=str, default='small')
    parser.add_argument('--layer_ids_set', type=str, default='small')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--model', type=str, default='knn_11')
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--handle_nan', type=str, default=None)


    H.update(parser.parse_args(s).__dict__)
    return H

def setup(H):
    cols = get_attributes(H.keys_set)

    if H.layer_ids_set == "small":
        latent_ids = [0,1,2,3,5,10,15,20,30,40,50]
    elif H.layer_ids_set == "mid":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53, 58, 63]
    elif H.layer_ids_set == "mid_cuda":
        latent_ids = list(np.arange(0, 42, 1)) + [48, 53, 58] # layer 43/44 is too large for cuKNeighborsClassifier (doesn't fit gpu)
    elif H.layer_ids_set == "full":
        latent_ids = get_available_latents()
    else:
        raise ValueError(f"Unknown latent ids set {H.layer_ids_set}")

    logging.basicConfig(level=H.log_level)

    return cols, latent_ids

def main():
    H = parse_args()
    cols, latent_ids = setup(H)

    wandb.run.name = H.run_name if H.run_name else f"{H.model}_{H.keys_set}_{H.layer_ids_set}"
    wandb.run.save()

    wandb.config.update(H)
    wandb.config.update({"cols": cols, "latent_ids": latent_ids})
    path = wandb.run.dir

    wandb.save("*.csv")

    # no wandb
    # path = f"outputs/"
    # if not os.path.exists(path):
    #     os.makedirs(path)

    logging.info(cols)
    logging.info(latent_ids)

    # scores = defaultdict(list)
    use_cuda = len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0
    for it, i in enumerate(tqdm(latent_ids)):
        try:
            score_dict = run_classifications(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan, cuda=use_cuda)
        except Exception as e:
            if use_cuda == True:
                logging.warning(f"While running on GPU caught {e}")
                logging.warning("trying without CUDA")
                use_cuda = False
                score_dict = run_classifications(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan,
                                                 cuda=use_cuda)
            else:
                raise e


        if it == 0:
            score_keys = list(score_dict[cols[0]].keys())

        for col in cols:
            fpath = os.path.join(path, f"{col}.csv")
            if it == 0:
                with open(fpath, "w") as fh:
                    fh.write(",".join(score_keys))

            results = [str(score_dict[col][k]) for k in score_keys]
            with open(fpath, "a") as fh:
                fh.write(",".join(results))

        # for col, score in score_dict.items():
        #     scores[col].append(score)
    #
    # for col in cols:
    #     df = pd.DataFrame(scores[col])
    #     df.to_csv(os.path.join(path, f"{col}.csv"), index=False)


if __name__ == "__main__":
    main()