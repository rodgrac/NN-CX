import torch
import numpy as np
import torch.nn as nn
import pytest

from nncx.tensor import Tensor
from nncx.layers.pooling import MaxPool2d
from nncx.enums import BackendType, DataType


# Globals
rtol = 1e-5
atol = 1e-5


@pytest.mark.parametrize("bsz, ch, H, W, ksize, stride",[
    (1, 3, 4, 4, 2, 2),
    (2, 1, 6, 6, 3, 2)
])
def test_pooling(bsz, ch, H, W, ksize, stride):
    np.random.seed(42)
    torch.manual_seed(42)
    
    backend_type = BackendType.CPU
    
    x = Tensor(shape=(bsz, ch, H, W), dtype=DataType.FLOAT32, backend_type=backend_type, grad_en=True).rand(-1, 1)
        
    pool = MaxPool2d(backend_type, kernel_size=ksize, stride=stride)
    
    y = pool(x)
    y.backward()
        
    ######### TORCH ###########    
    x_tor = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
    
    pool_tor = nn.MaxPool2d(kernel_size=ksize, stride=stride)

    y_tor = pool_tor(x_tor)
    y_tor.backward(gradient=torch.ones_like(y_tor))
    
    
    ######### EVAL ##############
    np.testing.assert_allclose(y.data, y_tor.detach().numpy(), rtol=rtol, atol=atol)

    # grad
    np.testing.assert_allclose(x.grad, x_tor.grad.detach().numpy(), rtol=rtol, atol=atol)
    
    
