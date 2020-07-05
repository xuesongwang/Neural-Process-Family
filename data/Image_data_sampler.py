import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CelebA, ImageFolder
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os



def train_val_split(trainset, split_ratio = 0.1):
    """
    Split original training set into training and validating sets
    Args:
        trainset: oiginal dataset
        split_ratio: len(validation set) / len(original dataset), default: 3:10
    Returns:
        a new training and a validation index samplers
    """
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(split_ratio * num_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler


class ImageReader(object):
    """Load image datasets from torchvision.datasets
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 datapath,
                 testing=False,
                 device = torch.device("cpu")):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
        dataset: dataset name, "MNIST" or "SVHN" or "celebA"
        batch_size: An integer.
        """
        self.dataset = dataset
        self._batch_size = batch_size
        self._testing = testing
        self.device = device

        if dataset == 'MNIST':
            trainset = MNIST(os.path.join(datapath,'./MNIST/mnist_data'), train=True, download=False,
                             transform=tf.ToTensor())
            testset = MNIST(os.path.join(datapath,'./MNIST/mnist_data'), train=False, download=False,
                            transform=tf.ToTensor())
            trainsampler, validsampler = train_val_split(trainset)
            trainloader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler, shuffle=False,
                                     num_workers=8)
            valloader = DataLoader(trainset, batch_size=batch_size, sampler=validsampler, shuffle=False)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        elif dataset == 'SVHN':
            trainset = SVHN(os.path.join(datapath,'./SVHN'), split='train', download=False,
                            transform=tf.ToTensor())
            testset = SVHN(os.path.join(datapath,'./SVHN'), split='test', download=False,
                           transform=tf.ToTensor())
            trainsampler, validsampler = train_val_split(trainset)
            trainloader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler, shuffle=False,
                                     num_workers=8)
            valloader = DataLoader(trainset, batch_size=batch_size, sampler=validsampler, shuffle=False)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        elif dataset == 'celebA':   # celebA datasets in torchvision build in validation datasets.
            transform = tf.Compose([
                tf.Resize([32, 32]),
                tf.ToTensor(),
            ])
            trainset = ImageFolder(os.path.join(datapath,'./celebA/train/'), transform)
            trainsampler, validsampler = train_val_split(trainset)
            trainloader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler, shuffle=False,
                                     num_workers=8, drop_last=True)
            valloader = DataLoader(trainset, batch_size=batch_size, sampler=validsampler, shuffle=False,
                                   drop_last=True)
            testset = ImageFolder(os.path.join(datapath,'./celebA/test/'), transform)
            testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=8,
                                    drop_last=True)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader


