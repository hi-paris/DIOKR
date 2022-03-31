import pytest
import numpy as np
import torch
from os.path import join
from DIOKR.DIOKR.utils import project_root
from skmultilearn.dataset import load_from_arff
import copy
from DIOKR.DIOKR import kernel, cost
from DIOKR.DIOKR.kernel import linear_kernel, rbf_kernel, gaussian_tani_kernel

path_tr = join(project_root(), 'data/bibtex/bibtex-train.arff')
x_train, y_train = load_from_arff(path_tr, label_count=159)
x_train, y_train = x_train.todense(), y_train.todense()

path_te = join(project_root(), 'data/bibtex/bibtex-test.arff')
x_test, y_test = load_from_arff(path_te, label_count=159)
x_test, y_test = x_test.todense(), y_test.todense()

x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

n = x_train.shape[0]

gamma_input = 0.1
gamma_output = 1.0

kernel_output = kernel.Gaussian(gamma_output)
lbda = 0.01

batch_size_train = 256

cost_function = cost.sloss_batch

d_out = int(x_train.shape[1] / 2)
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], d_out),
    torch.nn.ReLU(),
)

X_kernel = torch.normal(0, 1, size=(3, 1))
Y_kernel = torch.normal(0, 1, size=(2, 1)),

optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)


class TestLinearKernel():
    "Class test for linear Kernel function"

    def test_output_good_type_and_exists(self):
        K = linear_kernel(X_kernel)
        #print("XK:",X_kernel)
        #print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"


class TestRbfKernel():
    "Class test for rbf Kernel function"

    def test_output_good_type_and_exists(self):
        K = rbf_kernel(X_kernel)
        #print("XK:",X_kernel)
        #print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

class TestGaussianTaniKernel():
    "Class test for rbf Kernel function"

    def test_output_good_type_and_exists(self):
        K = gaussian_tani_kernel(X_kernel)
        #print("XK:",X_kernel)
        #print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

class TestGaussianComputeGram():
    "Class test for Gaussian.compute_gram function"

    def test_output_good_type_and_exists(self):
        kernel_output = kernel.Gaussian(gamma_output)
        K = kernel_output.compute_gram(X=X_kernel)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

class TestGaussianTaniComputeGram():
    "Class test for Gaussian.compute_gram function"

    def test_output_good_type_and_exists(self):
        kernel_output = kernel.GaussianTani(gamma_output)
        K = kernel_output.compute_gram(X=X_kernel)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

class TestLearnableGaussian():

    def test_compute_gram_output_good_type_and_exists(self):
        kernel_output = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
        K = kernel_output.compute_gram(X=x_train)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_compute_gram_frozen_output_good_type_and_exists(self):
        kernel_output = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
        K = kernel_output.compute_gram_frozen(X=x_train, Y=y_train)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_clone_kernel_output_same_as_origin(self):
        origin_kernel = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
        cloned_kernel = origin_kernel.clone_kernel()
        #assert cloned_kernel.optim_params == kernel_output.optim_params
        #assert cloned_kernel.model == kernel_output.model
        assert type(cloned_kernel) == kernel.LearnableGaussian, f"type of 'cloned_kernel' should be the same as 'origin_kernel"


class TestLearnableLinear():

    def test_compute_gram_output_good_type_and_exists(self):
        kernel_output = kernel.LearnableLinear(model_kernel_input, optim_params)
        K = kernel_output.compute_gram(X=x_train)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_compute_gram_frozen_output_good_type_and_exists(self):
        kernel_output = kernel.LearnableLinear(model_kernel_input, optim_params)
        K = kernel_output.compute_gram_frozen(X=X_kernel, Y=Y_kernel)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_clone_kernel_output_same_type_as_origin(self):
        origin_kernel = kernel.LearnableLinear(model_kernel_input, optim_params)
        cloned_kernel = origin_kernel.clone_kernel()
        #assert cloned_kernel.optim_params == kernel_output.optim_params
        #assert cloned_kernel.model == kernel_output.model
        assert type(cloned_kernel) == kernel.LearnableLinear, f"type of 'cloned_kernel' should be the same as 'origin_kernel"
