#import pytest
import torch
import torch.optim as optim
from DIOKR.DIOKR import cost
from DIOKR.DIOKR import kernel
from DIOKR.DIOKR import IOKR
from DIOKR.DIOKR import estimator
from scipy.linalg import block_diag
from os.path import join
from DIOKR.DIOKR.utils import project_root
from skmultilearn.dataset import load_from_arff

dtype = torch.float

#DATA
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

#OTHER
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

optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)
kernel_input = kernel.LearnableGaussian(
    gamma_input, model_kernel_input, optim_params)

iokr = IOKR()

diokr_estimator = estimator.DIOKREstimator(kernel_input, kernel_output,
                                           lbda, iokr=iokr, cost=cost_function)

#diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test, n_epochs=20, solver='sgd', batch_size_train=batch_size_train)

#obj = diokr_estimator.objective(x_batch = x_train , y_batch = y_train)

class TestObjective():

    def test_objective_returns_good_type_and_size(self):
        obj = diokr_estimator.objective(x_batch = x_train , y_batch = y_train)
        assert type(obj) == torch.Tensor, f"'obj' should be of type 'torch.Tensor', but is of type {type(obj)}"
        #assert obj.Size() == ,

class TestTrainKernelInput():

    def test_train_kernel_input_returns_good_type(self):
        mse_train = diokr_estimator.train_kernel_input(x_batch=x_train,
                                                       y_batch=y_train,
                                                       solver='sgd',
                                                       t0=0)
        assert type(mse_train) == float, f"'mse_train' should be a float, but is {type(mse_train)} instead"
        assert mse_train >= 0, f"'mse_train cannot be negative, and is of {mse_train}"

class TestFitKernelInput():

   def test_verbose(self, capfd):
         """Test if fit function actually prints something
         Parameters
         ----------
         capfd: fixture
         Allows access to stdout/stderr output created
         during test execution.
         Returns
         -------
         None
         """

         """Test if fit function actually prints something"""
         scores = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                 n_epochs=20, solver='sgd', batch_size_train=batch_size_train)
         out, err = capfd.readouterr()
         assert err == "", f'{err}: need to be fixed'
         assert out != "", f'Each Epoch run and MSE should have been printed'

class TestPredict():

    def test_predict_returns_good_type_and_shape(self):
        diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                         n_epochs=10, solver='sgd', batch_size_train=batch_size_train)
        Y_pred_test = diokr_estimator.predict(x_test=x_test)
        assert type(Y_pred_test) == torch.Tensor, f"'Y_pred' should be a torch.Tensor, but is {type(Y_pred_test)}"
        assert Y_pred_test.size() == (x_test.shape[0], y_test.shape[1]), f"Wrong shape for 'Y_pred_test':" \
                                                                         f"should be: ({x_test.shape[0]}, {y_test.shape[1]})" \
                                                                         f"Is: {Y_pred_test.size()}"