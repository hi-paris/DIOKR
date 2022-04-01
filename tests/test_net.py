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

#dim_inputs = x_train.shape[1]
#dim_outputs = int(dim_inputs/2)
#x = net.Net1(dim_inputs, dim_outputs)
#x = x.get_layers()
#print(x)
#print(type(x))



class TestNet1():

    def test_net_instance_good_shape(self, capfd):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self, ):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"

class TestNet2():

    def test_net_instance_good_shape(self, capfd):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"

class TestNet3():

    def test_net_instance_good_shape(self, capfd):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self, ):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"