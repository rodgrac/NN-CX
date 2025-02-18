import torch
import torch.nn as nn
import numpy as np

from nncx.tensor import Tensor
from nncx.layers.linear import Linear


def test_grad_linear():
    x = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    
    linear_ = Linear(5, 10)
    
    y = linear_(x)
    
    y.backward()
    
    ######### TORCH ###########    
    x_tor = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
    
    linear_tor_ = nn.Linear(5, 10)
    
    y_tor = linear_tor_(x_tor)
    y_tor.backward(gradient=torch.ones_like(y_tor))
        
    assert np.allclose(linear_.w.grad.T, linear_tor_.weight.grad.detach().numpy(), atol=1e-6)
    
