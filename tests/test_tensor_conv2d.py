import torch
import numpy as np
import torch.nn as nn
import pytest

from nncx.tensor import Tensor
from nncx.layers.conv2d import Conv2d
from nncx.enums import BackendType, DataType


# Globals
rtol = 1e-5
atol = 1e-5


@pytest.mark.parametrize("bsz, in_ch, out_ch, H, W, ksize, pad, stride",[
    (1, 3, 2, 5, 5, 3, 1, 1),
    (2, 1, 1, 8, 8, 3, 0, 2)
])
def test_conv2d(bsz, in_ch, out_ch, H, W, ksize, pad, stride):
    np.random.seed(42)
    torch.manual_seed(42)
    
    backend_type = BackendType.CPU
    
    x = Tensor(shape=(bsz, in_ch, H, W), dtype=DataType.FLOAT32, backend_type=backend_type, grad_en=True).rand(-1, 1)
        
    conv = Conv2d(in_ch, out_ch, backend_type, ksize, stride, pad, bias=True)
    
    y = conv(x)
    y.backward()
        
    ######### TORCH ###########    
    x_tor = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
    
    conv_tor = nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding='same' if pad else 'valid', bias=True)
    
    # copy weights/bias
    with torch.no_grad():
        conv_tor.weight.copy_(torch.tensor(conv.w.data))
        conv_tor.bias.copy_(torch.tensor(conv.bias.data))

    y_tor = conv_tor(x_tor)
    y_tor.backward(gradient=torch.ones_like(y_tor))
    
    
    ######### EVAL ##############
    np.testing.assert_allclose(y.data, y_tor.detach().numpy(), rtol=rtol, atol=atol)

    # grad
    np.testing.assert_allclose(conv.w.grad, conv_tor.weight.grad.detach().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(conv.bias.grad, conv_tor.bias.grad.detach().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(x.grad, x_tor.grad.detach().numpy(), rtol=rtol, atol=atol)
    
    
