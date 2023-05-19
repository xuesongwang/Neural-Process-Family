"""
author:  YannDubs
https://github.com/YannDubs/Neural-Process-Family/blob/master/utils/data/gaussian_process.py
"""
import os
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)
import torch
import numpy as np
import sys
sys.path
sys.path.append('data')
from GP_dataset_NPF import GPDataset, DIR_DATA


def get_gp_dataset(name='RBF', mode='train', n_samples=50000):
    # define Kernel by name
    if name == 'RBF':
        kernel = RBF(length_scale=(0.2))
    elif name == "Periodic_Kernel":
        kernel = ExpSineSquared(length_scale=0.5, periodicity=0.5)
    elif name == 'Matern': # Matern 3/2
        kernel = Matern(length_scale=0.2, nu=1.5)

    # generate sample and save
    save_file = f"{os.path.join(DIR_DATA, 'gp_dataset_%s_%s.hdf5'%(name, mode))}"

    dataset = GPDataset(kernel=kernel, save_file=save_file, n_samples =n_samples)
    # return dataset
    return dataset

def collate_fns(max_num_context, max_num_extra_target):
    def collate_fn(batch):
        # Collate
        x = torch.stack([x for x, _, _ in batch], 0)
        y = torch.stack([y for _, y, _ in batch], 0)
        kernel_index = torch.tensor([idx for _, _, idx in batch],requires_grad=False)

        # Sample a subset of random size
        num_context = np.random.randint(3, max_num_context)
        num_extra_target = np.random.randint(3, max_num_extra_target)

        inds = np.random.choice(range(x.shape[1]), size=(num_context + num_extra_target), replace=False)
        context_x = x[:, inds][:, :num_context]
        context_y = y[:, inds][:, :num_context]

        target_x = x[:, inds][:, num_context:]
        target_y = y[:, inds][:, num_context:]
        return context_x, context_y, target_x, target_y, kernel_index
    return collate_fn

def gp_transport_task(kernel1='RBF', kernel2 = 'Periodic_Kernel'):
    scale = 1 if torch.cuda.is_available() else 0.002
    n_samples = int(50000*scale)
    tau_dataset = get_gp_dataset(name=kernel1, mode='tau', n_samples=n_samples)

    dataset_tr_k1 = get_gp_dataset(name=kernel1, mode='tr', n_samples=n_samples)
    dataset_tr_k2 = get_gp_dataset(name=kernel2, mode='tr', n_samples=n_samples)

    # I want to create a mixed_training set with half of the dataset_tr_k1/2 so that the mixed data size is the same as dataset_tr_k1
    mixed_tr_k1, _ = torch.utils.data.random_split(dataset_tr_k1, [0.5, 0.5])
    mixed_tr_k2, _ = torch.utils.data.random_split(dataset_tr_k2, [0.5, 0.5])
    mixed_tr = torch.utils.data.ConcatDataset([mixed_tr_k1, mixed_tr_k2])

    dataset_val_k1 = get_gp_dataset(name=kernel1, mode='val', n_samples=int(1000*scale))
    dataset_val_k2 = get_gp_dataset(name=kernel2, n_samples=int(1000*scale))
    return tau_dataset, dataset_tr_k1, dataset_tr_k2, mixed_tr, dataset_val_k1, dataset_val_k2

if __name__ == '__main__':
    get_gp_dataset(name='RBF', )
    get_gp_dataset(name='Periodic_Kernel')