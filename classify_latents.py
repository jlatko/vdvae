import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from latents import get_latents
# TODO: ?
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    z, meta = get_latents(latents_dir="/scratch/s193223/vdvae/latents/", layer_ind=0, splits=[1])
    col = "Young"
    y = np.array(meta[col]==1)
    z = z.reshape(z.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.33, stratify=y, random_state=42)

    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=8)

    neigh.fit(X_train, y_train)

    print(neigh.score(X_test, y_test))

if __name__ == "__main__":
    main()