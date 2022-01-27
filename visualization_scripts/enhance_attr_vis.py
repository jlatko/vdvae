import argparse
import re
import traceback

import matplotlib
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import os

from vdvae.hps import Hyperparams
from vdvae.wandb_utils import _download

api = wandb.Api()

fig, ax = plt.subplots()
DPI = fig.dpi
plt.close()


project_viz="johnnysummer/vae_visualizations"
project_scores = "johnnysummer/vdvae_analysis"

def get_scores(H, run_scores, attr):
    files_scores = run_scores.files()
    name2file_scores = {f.name: f for f in files_scores}
    if f'{attr}.csv' not in name2file_scores:
        print(f"{attr}.csv not found in run {run_scores.id}")
        return None
    path_score = _download(name2file_scores[f'{attr}.csv'], f"./.data/{run_scores.id}/")
    scores = pd.read_csv(path_score)
    scores = scores.set_index("layer_ind")
    return scores

def get_img(H, run_viz, f):
    lv_points = run_viz.config['lv_points']
    path = _download(f, f"./.data/{H.run_id_viz}/")
    img = Image.open(path)
    return img, lv_points

def enhance_attribute_visualization(H, file, runs_scores, run_viz,
                                    temp=0.1, size=64):
    attr = re.match(r"([a-zA-Z_]+)_t.*", file.name).group(1)
    scores = [get_scores(H, run, attr) for run in runs_scores]
    scores = [s for s in scores if s is not None]
    if len(scores) == 0:
        print(f"No scores for {attr}")
        scores = [pd.DataFrame()]
    img, lv_points = get_img(H, run_viz, file)

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 8]},
                               figsize=(1.5 * img.width / DPI, 1.1 * img.height / DPI))

    a1.imshow(img, aspect=1)
    plt.title(f"{attr} (t={temp})", fontsize=24)

    scores_picked = [
        [f"{df.loc[i, H.scores_key]:.3f}" if i in df.index else "?" for i in lv_points]
        for df in scores
    ]
    res_picked = [scores[0].loc[i, 'resolution'] if i in scores[0].index else "?" for i in lv_points]
    yticks = [
        f"{i}\n({s})\n{res}x{res}"
        for i, s, res in zip(lv_points, scores_picked[0], res_picked)
    ]
    plt.sca(a1)
    plt.yticks(size / 2 + size * np.arange(len(lv_points)), yticks)
    plt.xticks([size / 2, img.width - size / 2], [f"Less {attr}", f"More {attr}"])
    plt.tick_params(axis='both', labelsize=18, length=0)

    plt.sca(a0)
    a0.set_xlim((0.45, 0.8))

    a0.invert_xaxis()
    a0.spines['top'].set_visible(False)
    # a0.spines['right'].set_visible(False)
    # a0.spines['bottom'].set_visible(False)
    a0.spines['left'].set_visible(False)
    # plt.axis('off')

    # sns.barplot(x=lv_points, y=scores_picked, ax=a0, orient='h', width=size/2/DPI)

    for j, df in enumerate(scores):
        scores_picked = [df.loc[i, H.scores_key] if i in df.index else 0 for i in lv_points]
        a0.barh(np.arange(len(lv_points))[::-1] * size / DPI - j * size / 3 / DPI,
                scores_picked,
                height= size / 4 / DPI,
                color=matplotlib.cm.get_cmap("cool")(scores_picked)
                )
    a0.set_ylim((-size / 2 / DPI, (len(lv_points) * size - size / 2) / DPI))
    plt.xlabel(H.scores_key)
    plt.title(','.join(run.config['model'] for run in runs_scores), fontsize=24)
    # yticks = [
    #     f"{i}"
    #     for i, s in zip(lv_points, scores_picked)
    # ]
    # plt.yticks(np.arange(len(lv_points))[::-1]*size/DPI, yticks);
    plt.yticks([], [])
    a0.yaxis.tick_right()
    # a0.xaxis.set_visible(False)

    plt.subplots_adjust(wspace=0.02)

    plt.savefig(os.path.join(wandb.run.dir, f"{file.name.split('.')[0]}_{H.scores_key}.jpg"), bbox_inches='tight')

    wandb.log({f"{attr}": wandb.Image(plt, caption=f"{file.name.split('.')[0]}_{H.scores_key}")})
    return scores



def parse_args(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default=None)
    # parser.add_argument('--size', type=int, default=128)
    # parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--run_id_viz', type=str, default=None)
    parser.add_argument('--run_id_scores', help='delimited list input',
                        type=lambda s: [item for item in s.split(',')], default=None)
    parser.add_argument('--scores_key', type=str, default="roc_auc_score_avg")


    H.update(parser.parse_args(s).__dict__)
    return H


def init_wandb(H, run_viz, runs_scores):
    tags = []
    tags.append(str(len(runs_scores)))
    if run_viz.config["grouped"]:
        tags.append("grouped")
    if run_viz.config["use_group_direction"]:
        tags.append("group_direction")
    if run_viz.config["has_attr"]:
        tags.append("has_attr")
    if run_viz.config["fixed"]:
        tags.append("fixed")

    if "latent_key" in run_viz.config:
        tags.append(run_viz.config["latent_key"])
    else:
        tags.append("z")

    # wandb.init(project='vae_visualizations', entity='johnnysummer', dir="/scratch/s193223/wandb/", tags=tags)
    wandb.init(project='vae_visualizations', entity='johnnysummer', tags=tags)
    wandb.config.update({"script": "enhance"})

    if H.run_name is None:
        wandb.run.name = "e_" + run_viz.name + "__" + '_'.join([run.config['model'] for run in runs_scores]) + "__" + wandb.run.name.split('-')[-1]
        wandb.run.save()
    else:
        wandb.run.name = "e_" + H.run_name + "-" + wandb.run.name.split('-')[-1]
        wandb.run.save()

def main():
    H = parse_args()


    run_viz = api.run(f"{project_viz}/{H.run_id_viz}")
    runs_scores = [
        api.run(f"{project_scores}/{run_id}")
        for run_id in H.run_id_scores]


    init_wandb(H, run_viz, runs_scores)

    temp = run_viz.config["temp"]
    size = run_viz.config["size"]

    print(run_viz.config)

    files = run_viz.files()
    name2file = {f.name: f for f in files}
    for f in name2file:
        if f.endswith(".png") and "/" not in f:
            print(f)
            try:
                enhance_attribute_visualization(H, name2file[f], runs_scores, run_viz, temp=temp, size=size)
            except Exception as e:
                print(f"Caught error for {f}")
                print(e)

                traceback.print_exc()
                print("continuing...")

if __name__ == "__main__":
    main()
