import argparse
import os

from attributes import get_attributes
from wandb_utils import _download

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
import gc

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

MIN_FREQ = 100

# TODO: consider accounting for the most frequent and correlated attributes (earings only for females etc)

def get_model(H, cuda):
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
            model = SVC()
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

    return model

def get_classification_score(H, X_train, X_test, y_train, y_test, cuda=False):
    # TODO: normalize??
    model = get_model(H, cuda)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    del model
    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc_score': roc_auc_score(y_test, y_pred),
    }

def run_folds(H, z, meta, col, cuda, layer_ind, prefix=""):
    kfold = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    kfold_scores = []
    y = np.array(meta[col] == 1)
    if y.sum() < MIN_FREQ or (~y).sum() < MIN_FREQ:
        logging.info(f"skipping {col}")
        return None
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
    score["frequency"] = y.sum() / len(y)
    log_score(score, col, layer_ind, prefix=prefix)
    return score

def get_all_scores(H, z, meta, cols, cuda, layer_ind, prefix=""):
    scores = {}
    # z = cudf.DataFrame(z)
    for col in cols:
        score = run_folds(H, z, meta, col, cuda, layer_ind, prefix=prefix)
        scores[col] = score
    return scores

def log_score(score, col, layer_ind, prefix=""):
    wandb.log({
        f"{prefix}{col}_{k}": v
        for k, v
        in score.items() if k not in ["len", "size", "resolution", "layer_ind"]
    }, step=layer_ind)
    logging.debug(f"{prefix}{col}: {score}")


def group_scores(scores_male, scores_female, cols, layer_ind):
    scores = {}

    for col in cols:
        scores[col] = {}
        if scores_male.get(col) is not None:
            for k, v in scores_male[col].items():
                scores[col][f"m_{k}"] = v
            if scores_female.get(col) is None:
                for k, v in scores_male[col].items():
                    scores[col][k] = v

        if scores_female.get(col) is not None:
            for k, v in scores_female[col].items():
                scores[col][f"f_{k}"] = v
            if scores_male.get(col) is None:
                for k, v in scores_female[col].items():
                    scores[col][k] = v

        if (scores_female.get(col) is not None) and (scores_male.get(col) is not None):
            for k in scores_male[col].keys():
                scores[col][k] =  (scores_male[col][k] +  scores_female[col][k]) / 2

                wandb.log({
                    f"{col}_{k}": scores[col][k]
                }, step=layer_ind)

    return scores


def run_classifications(H, cols, layer_ind, latents_dir, handle_nan=False, cuda=False, previous=None):
    if previous is None:
        previous = {}
    # TODO: get from previous and skip?
    cols_filtered = set(cols)
    scores = {}
    for col in cols:
        if col in previous:
            if layer_ind in list(previous[col].layer_ind):
                scores[col] = previous[col][previous[col].layer_ind == layer_ind].iloc[0].to_dict()
                log_score(scores[col], col, layer_ind, prefix="")
                cols_filtered = cols_filtered - {col}

    if len(cols_filtered) == 0:
        logging.info(f"Found all scores for layer {layer_ind}. Skipping.")
        return scores

    z, meta = get_latents(latents_dir=latents_dir, layer_ind=layer_ind, splits=H.splits, allow_missing=False, handle_nan=handle_nan)
    logging.debug(z.shape)

    resolution = z.shape[-2]
    wandb.log({"resolution": resolution}, step=layer_ind)
    z = z.reshape(z.shape[0], -1)
    wandb.log({"size": z.shape[1]}, step=layer_ind)

    z_info = {}
    z_info["len"] = z.shape[0]
    z_info["size"] = z.shape[1]
    z_info["resolution"] = resolution
    z_info["layer_ind"] = layer_ind

    # TODO: ? move to gpu? X_cudf = cudf.DataFrame(X)

    if H.grouped:
        q = meta["Male"] == 1
        scores_male = get_all_scores(H, z[q], meta[q], cols_filtered, cuda, layer_ind, prefix="m_")
        scores_female = get_all_scores(H, z[~q], meta[~q], cols_filtered, cuda, layer_ind, prefix="f_")
        scores.update(group_scores(scores_male, scores_female, cols_filtered, layer_ind))
    else:
        scores.update(get_all_scores(H, z, meta, list(cols_filtered), cuda, layer_ind))

    for k in cols_filtered:
        if k in scores:
            scores[k].update(z_info)

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
    parser.add_argument('--cont_run', type=str, default=None)
    parser.add_argument('--grouped', action="store_true")
    parser.add_argument('-s', '--splits', help='delimited list input',
                        type=lambda s: [int(item) for item in s.split(',')], default=[1,2,3])


    H.update(parser.parse_args(s).__dict__)
    return H

def setup(H):
    cols = get_attributes(H.keys_set)

    if H.layer_ids_set == "small":
        latent_ids = [0,1,2,3,5,10,15,20,30,40]
    elif H.layer_ids_set == "mid":
        latent_ids = list(range(11)) + list(np.arange(12, 21, 2)) + list(np.arange(21, 42, 3)) + [43, 48, 53, 58, 63]
    elif H.layer_ids_set == "mid_cuda":
        latent_ids = list(np.arange(0, 42, 1)) + [43, 48, 53, 58] # layer 43/44 is too large for cuKNeighborsClassifier (doesn't fit gpu)
    elif H.layer_ids_set == "full":
        latent_ids = get_available_latents()
    else:
        raise ValueError(f"Unknown latent ids set {H.layer_ids_set}")

    logging.basicConfig(level=H.log_level)

    return cols, latent_ids

def load_previous(H):
    if H.cont_run is None:
        return {}
    api = wandb.Api()
    run = api.run(f"vdvae_analysis/{H.cont_run}")
    files = run.files()
    name2file = {f.name: f for f in files if f.name.endswith(".csv")}
    attr2csv = {}
    for fname, file in name2file.items():
        attr = fname.split(".")[0]
        path = _download(file, f"./.data/{H.cont_run}/")
        df = pd.read_csv(path)
        attr2csv[attr] = df
    return attr2csv

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
    if H.grouped:
        cols = [c for c in cols if c != "Male"] # filter out Male as we group by sex

    logging.info(cols)
    logging.info(latent_ids)

    previous = load_previous(H)

    # scores = defaultdict(list)
    use_cuda = len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0
    for it, i in enumerate(tqdm(latent_ids)):
        try:
            score_dict = run_classifications(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan, cuda=use_cuda, previous=previous)
        except Exception as e:
            if use_cuda == True:
                logging.warning(f"While running on GPU caught {e}")
                logging.warning("trying without CUDA")
                gc.collect()
                use_cuda = False
                score_dict = run_classifications(H, cols, i, latents_dir=H.latents_dir, handle_nan=H.handle_nan,
                                                 cuda=use_cuda, previous=previous)
            else:
                raise e

        if it == 0:
            score_keys = set()
            for col in cols:
                score_keys = score_keys | set(score_dict[col].keys())
            score_keys = list(score_keys)

        for col in cols:
            if col in score_dict:
                fpath = os.path.join(path, f"{col}.csv")
                if it == 0:
                    with open(fpath, "w") as fh:
                        fh.write(",".join(score_keys))

                results = [str(score_dict[col][k]) if k in score_dict[col] else "" for k in score_keys]
                with open(fpath, "a") as fh:
                    fh.write("\n" + ",".join(results))
            else:
                logging.warning(f"{col} missing for layer {i}")
        del score_dict
        gc.collect()
        # for col, score in score_dict.items():
        #     scores[col].append(score)
    #
    # for col in cols:
    #     df = pd.DataFrame(scores[col])
    #     df.to_csv(os.path.join(path, f"{col}.csv"), index=False)


if __name__ == "__main__":
    main()