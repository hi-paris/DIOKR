"""
import pytest
from DIOKR.DIOKR import net
import torch
import numpy as np
import torch.optim as optim
from DIOKR.DIOKR import cost
from DIOKR.DIOKR import kernel
from DIOKR.DIOKR import IOKR
from DIOKR.DIOKR import estimator
from scipy.linalg import block_diag
from os.path import join
from DIOKR.DIOKR.utils import project_root
from skmultilearn.dataset import load_from_arff

path_tr = join(project_root(), 'data/bibtex/bibtex-train.arff')
x_train, y_train = load_from_arff(path_tr, label_count=159)
x_train, y_train = x_train.todense(), y_train.todense()

x_train, y_train = x_train[:2000], y_train[:2000]

x_train = torch.from_numpy(x_train).float()

dim_inputs = x_train.shape[1]
dim_outputs = dim_inputs/2

x = net.Net1(dim_inputs, dim_outputs)

print(x)
print(type(x))
"""