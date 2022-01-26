from PIL import Image
import numpy as np
from tqdm import tqdm

res = 64
SOURCE_PATH = "/scratch/s193223/ffhq/ffhq-256.npy"
DESTINATION = f"/scratch/s193223/ffhq/ffhq-{res}npy"
trX = np.load((SOURCE_PATH), mmap_mode='r')
L = trX.shape[0]
new_X = np.zeros([L, res, res, 3], dtype='uint8')
for i in tqdm(range(L)):
    img = Image.fromarray(trX[i])
    img_new = img.resize((res, res))
    x = np.array(img_new)
    new_X[i,:,:,:] = x
np.save(DESTINATION, new_X, allow_pickle=False)