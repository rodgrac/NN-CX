import torch
import numpy as np
import torch.nn.functional as F
import pytest

from nncx.backend.utils import init_backend
from nncx.tensor import Tensor
from nncx.layers.conv2d import Conv2d
from nncx.enums import BackendType, DataType


@pytest.mark.parametrize("bsz, in_ch, out_ch, H, W, ksize, pad, stride",[
    (1, 3, 2, 5, 5, 3, 1, 1),
    (2, 1, 1, 8, 8, 3, 0, 2)
])
def test_conv2d_forward(bsz, in_ch, out_ch, H, W, ksize, pad, stride):
    np.random.seed(42)
    torch.manual_seed(42)
    
    backend = init_backend(BackendType.CPU)
    
    x = Tensor(shape=(bsz, in_ch, H, W), dtype=DataType.FLOAT32, backend=backend, grad_en=True).rand(-1, 1)
        
    conv = Conv2d(in_ch, out_ch, backend, ksize, stride, pad, bias=True)
    
    y = conv(x)
        
    ######### TORCH ###########    
    x_tor = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
    y_tor = F.conv2d(x_tor, torch.from_numpy(conv.w.data), torch.from_numpy(conv.bias.data[0]),
                        stride=stride, padding='valid')
    
    np.testing.assert_allclose(y.data, y_tor.detach().numpy(), rtol=1e-4, atol=1e-4)
    

    
