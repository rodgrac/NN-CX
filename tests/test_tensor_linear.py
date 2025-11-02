import torch
import torch.nn as nn
import numpy as np
import pytest

from nncx.tensor import Tensor
from nncx.layers.linear import Linear
from nncx.enums import BackendType, DataType

# Globals
rtol = 1e-5
atol = 1e-5

@pytest.mark.parametrize("bsz, in_ftrs, out_ftrs",[
    (1, 5, 10),
    (2, 16, 8)
])
def test_grad_linear(bsz, in_ftrs, out_ftrs):
    np.random.seed(42)
    torch.manual_seed(42)
    
    backend_type = BackendType.CPU
    
    x = Tensor(shape=(bsz, in_ftrs), dtype=DataType.FLOAT32, backend_type=backend_type, grad_en=True).rand(-1, 1)
    
    linear = Linear(in_ftrs, out_ftrs, backend_type=backend_type)
    
    y = linear(x)
    y.backward()
    
    ######### TORCH ###########    
    x_tor = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
    
    linear_tor = nn.Linear(in_ftrs, out_ftrs)
    
    # copy weights/bias
    with torch.no_grad():
        linear_tor.weight.copy_(torch.tensor(linear.w.data))
        linear_tor.bias.copy_(torch.tensor(linear.bias.data))
    
    y_tor = linear_tor(x_tor)
    y_tor.backward(gradient=torch.ones_like(y_tor))
    
    ######### EVAL ##############
    np.testing.assert_allclose(y.data, y_tor.detach().numpy(), rtol=rtol, atol=atol)
    
    # grad
    np.testing.assert_allclose(linear.w.grad, linear_tor.weight.grad.detach().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(linear.bias.grad, linear_tor.bias.grad.detach().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(x.grad, x_tor.grad.detach().numpy(), rtol=rtol, atol=atol)