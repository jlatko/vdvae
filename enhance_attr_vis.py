import argparse

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import os

from attributes import get_attributes
from hps import Hyperparams

api = wandb.Api()

fig, ax = plt.subplots()
DPI = fig.dpi
plt.close()


project_viz="johnnysummer/vae_visualizations"
project_scores = "johnnysummer/vdvae_analysis"

def _download(file, path, force_redownload=False):
    full_path = os.path.join(path, file.name)
    if os.path.exists(full_path) and not force_redownload:
        return full_path
    else:
        file.download(path, replace=True)
        return full_path

def download_file(run_id, filename, project="johhnysummer/vdvae_analysis", force_redownload=False):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    files = run.files()
    for file in files:
        if file.name == filename:
            return _download(
                file, f"./data/{run_id}/", force_redownload=force_redownload
            )

def enhance_attribute_visualization(H, attr, run_scores, run_viz,
                                    temp=0.1, size=64):


    files_scores = run_scores.files()
    name2file_scores = {f.name: f for f in files_scores}
    path_score = _download(name2file_scores[f'{attr}.csv'], f"./data/{H.run_id_scores}/")
    scores = pd.read_csv(path_score)
    scores = scores.set_index("layer_ind")

    files = run_viz.files()
    name2file = {f.name: f for f in files}
    lv_points = run_viz.config['lv_points']
    path = _download(name2file[f'{attr}_t{str(temp).replace(".", "_")}_2.png'], f"./data/{H.run_id_viz}/")
    img = Image.open(path)

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 8]},
                               figsize=(1.5 * img.width / DPI, 1.1 * img.height / DPI))

    a1.imshow(img, aspect=img.height / img.width)
    plt.title(f"{attr} (t={temp})", fontsize=24)

    scores_picked = [f"{scores.loc[i, H.scores_key]:.3f}" if i in scores.index else "?" for i in lv_points]
    yticks = [
        f"{i}\n({s})"
        for i, s in zip(lv_points, scores_picked)
    ]
    plt.sca(a1)
    plt.yticks(size / 2 + size * np.arange(len(lv_points)), yticks);
    plt.xticks([size / 2, img.width - size / 2], [f"More {attr}", f"Less {attr}"]);
    plt.tick_params(axis='both', labelsize=20, length=0);

    scores_picked = [scores.loc[i, H.scores_key] if i in scores.index else 0 for i in lv_points]
    a0.set_xlim((0.45, 0.8))

    a0.invert_xaxis()
    a0.spines['top'].set_visible(False)
    # a0.spines['right'].set_visible(False)
    # a0.spines['bottom'].set_visible(False)
    a0.spines['left'].set_visible(False)
    # plt.axis('off')

    # sns.barplot(x=lv_points, y=scores_picked, ax=a0, orient='h', width=size/2/DPI)
    a0.barh(np.arange(len(lv_points))[::-1] * size / DPI,
            scores_picked,
            height=16 / DPI,
            color=matplotlib.cm.get_cmap("cool")(scores_picked)
            )
    a0.set_ylim((-size / 2 / DPI, (len(lv_points) * size - size / 2) / DPI))
    plt.sca(a0)
    plt.xlabel(H.scores_key)
    plt.title(run_scores.config['model'], fontsize=24)
    # yticks = [
    #     f"{i}"
    #     for i, s in zip(lv_points, scores_picked)
    # ]
    # plt.yticks(np.arange(len(lv_points))[::-1]*size/DPI, yticks);
    plt.yticks([], [])
    a0.yaxis.tick_right()
    # a0.xaxis.set_visible(False)

    plt.subplots_adjust(wspace=0.02)

    plt.savefig(os.path.join(wandb.run.dir, f"{attr}_{H.scores_key}.jpg"), bbox_inches='tight')

    return scores


wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/")
wandb.config.update({"script": "enhance"})

def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default=None)
    # parser.add_argument('--size', type=int, default=128)
    # parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--run_id_viz', type=str, default=None)
    parser.add_argument('--run_id_scores', type=str, default=None)
    parser.add_argument('--scores_key', type=str, default="roc_auc_score_avg")


    H.update(parser.parse_args(s).__dict__)
    return H

def main():
    H = parse_args()

    attributes = get_attributes(H.keys_set)

    run_viz = api.run(f"{project_viz}/{H.run_id_viz}")
    run_scores = api.run(f"{project_scores}/{H.run_id_scores}")
    temp = run_viz.config["temp"]
    size = run_viz.config["temp"]

    for attr in attributes:
        enhance_attribute_visualization(H, attr, run_scores, run_viz, temp=temp, size=size)


if __name__ == "__main__":
    main()