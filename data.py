import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from celebahq import CelebAHQDataset
from imagenet256 import ImageNet256Dataset
from noise_dataset import NoiseDataset


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
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
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

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

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
        inp = x[0].cuda(non_blocking=True).float()
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

def cifar10(data_root, one_hot=True):
    tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)
