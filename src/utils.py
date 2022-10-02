# encoding: utf-8

import torch
import numpy as np
from scipy.spatial.distance import cdist

# utils functions
def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(tensor, bound = 10.):
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def compute_adjacency_matrix_images(coord, sigma=0.01):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A
