import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import scipy.io as sio

from vdvae.data.celebahq import CelebAHQDataset
from vdvae.data.imagenet256 import ImageNet256Dataset
from vdvae.data.noise_dataset import NoiseDataset

CIFAR_GROUPS = {
    "animals": [2,3,4,5,6,7],
    "transportation": [0,1,8,9],
    "airplane" : [0],
    "automobile" : [1],
    "bird" : [2],
    "cat" : [3],
    "deer" : [4],
    "dog" : [5],
    "frog" : [6],
    "horse" : [7],
    "ship" : [8],
    "truck" : [9],
    "car_truck": [1, 9],
    "ship_airplane": [0,8],
    "cat_dog": [3,5],
    "deer_horse": [4, 7],
    "frog_bird": [2, 6]
}

def cuda(x, **kwargs):
    if torch.cuda.is_available():
        return x.cuda(**kwargs)
    else:
        return x

def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'imagenet64':
        trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'ffhq_256': # data (0,255)
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'ffhq_32': # data (0,255)
        trX, vaX, teX = ffhq32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        # like ffhq
        # shift = -112.8666757481
        # scale = 1. / 69.84780273
        # like cifar
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == 'ffhq_64': # data (0,255)
        trX, vaX, teX = ffhq64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'celebahq': # data (0,1)
        trX, vaX, teX = celebahq(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -0.4426144146984313 # same as ffhq256 * 255
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'i256': # data (0,1)
        trX, vaX, teX = None, None, None
        H.image_size = 256
        H.image_channels = 3
        shift = -0.4426144146984313 # same as celebahq
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False, group=H.cifar_group)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == 'svhn':
        trX, vaX, teX = svhn(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == 'gaussian_noise':
        trX, vaX, teX = None, None, None
        H.image_size = 256
        H.image_channels = 3
        shift = 0.
        scale = 1.
        shift_loss = 0.
        scale_loss = 0.33 # shouldn't
    elif H.dataset == 'uniform_noise':
        trX, vaX, teX = None, None, None
        H.image_size = 256
        H.image_channels = 3
        shift = 0.
        scale = 1.
        shift_loss = 0.
        scale_loss = 1. # shouldn't matter
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = cuda(torch.tensor([shift])).view(1, 1, 1, 1)
    scale = cuda(torch.tensor([scale])).view(1, 1, 1, 1)
    shift_loss = cuda(torch.tensor([shift_loss])).view(1, 1, 1, 1)
    scale_loss = cuda(torch.tensor([scale_loss])).view(1, 1, 1, 1)

    if H.dataset  == 'ffhq_1024':
        train_data = ImageFolder(trX, transforms.ToTensor())
        valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
        untranspose = True
    elif H.dataset == 'celebahq':
        train_data = CelebAHQDataset(root_dir=H.data_root,  train=True, transform=transforms.ToTensor(), splits=H.train_splits)
        valid_data = CelebAHQDataset(root_dir=H.data_root, train=False, transform=transforms.ToTensor(), splits=H.val_splits)
        untranspose = True
    elif H.dataset == 'i256':
        train_data = ImageNet256Dataset(transform=transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]))
        valid_data = train_data
        untranspose = True
    elif H.dataset == 'gaussian_noise':
        train_data = NoiseDataset(noise_type="gaussian")
        valid_data = NoiseDataset(noise_type="gaussian")
        untranspose = False
    elif H.dataset == 'uniform_noise':
        train_data = NoiseDataset(noise_type="uniform")
        valid_data = NoiseDataset(noise_type="uniform")
        untranspose = False
    else:
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))
        untranspose = False

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = cuda(x[0], non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        if do_low_bit:
            # 5 bits of precision
            out.mul_(1. / 8.).floor_().mul_(8.)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def imagenet32(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet32-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet32-valid.npy'), mmap_mode='r')
    return train, valid, test


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def celebahq(data_root):
    return os.path.join(data_root, 'img256train'), os.path.join(data_root, 'img256val'), os.path.join(data_root, 'img256val')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid

def ffhq64(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-64.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid

def ffhq32(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-32.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid


def cifar10(data_root, one_hot=True, group=None):
    tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)

    if group is not None:
        labels = CIFAR_GROUPS[group]

        print("Group", group, labels)
        print("Lengths before:", len(trY), len(vaY), len(teY), len(trY) + len(vaY) + len(teY))
        tr_mask = np.isin(trY, labels)
        va_mask = np.isin(vaY, labels)
        te_mask = np.isin(teY, labels)
        trX = trX[tr_mask]
        trY = trY[tr_mask]
        vaX = vaX[va_mask]
        vaY = vaY[va_mask]
        teX = teX[te_mask]
        teY = teY[te_mask]
        print("Lengths after:", len(trY), len(vaY), len(teY), len(trY) + len(vaY) + len(teY))


    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)


def svhn(data_root):
    trX = sio.loadmat(os.path.join(data_root, "train_32x32.mat"))["X"]
    teX = sio.loadmat(os.path.join(data_root, "test_32x32.mat"))["X"]
    trX = trX.transpose(3,0,1,2)
    teX = teX.transpose(3,0,1,2)
    trX, vaX = train_test_split(trX, test_size=5000, random_state=11172018)
    return trX, vaX, teX
