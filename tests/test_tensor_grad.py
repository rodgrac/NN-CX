import torch
import numpy as np

from nncx.tensor import Tensor

def test_grad_elementwise():
    ###### NNCL ########
    a = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    b = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    c = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    d = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    
    y = a * b + c**2 * (d - c)
    y.backward()
            
    ######### TORCH ###########    
    a_tor = torch.tensor(a.data, dtype=torch.float32, requires_grad=True)
    b_tor = torch.tensor(b.data, dtype=torch.float32, requires_grad=True)
    c_tor = torch.tensor(c.data, dtype=torch.float32, requires_grad=True)
    d_tor = torch.tensor(d.data, dtype=torch.float32, requires_grad=True)
    
    y_tor = a_tor * b_tor + c_tor**2 * (d_tor - c_tor)
    y_tor.backward(torch.ones_like(y_tor))    
        
    assert np.allclose(a.grad, a_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(b.grad, b_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(c.grad, c_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(d.grad, d_tor.grad.detach().numpy(), atol=1e-6)
    
    
def test_grad_matmul():
    a = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).randn()
    b = Tensor(shape=(5, 10), dtype=Tensor.FLOAT32, grad_en=True).randn()
    c = Tensor(shape=(10, 10), dtype=Tensor.FLOAT32, grad_en=True).randn()
    
    y = a @ b - c**2
    y.backward()
    
    ######### TORCH ###########    
    a_tor = torch.tensor(a.data, dtype=torch.float32, requires_grad=True)
    b_tor = torch.tensor(b.data, dtype=torch.float32, requires_grad=True)
    c_tor = torch.tensor(c.data, dtype=torch.float32, requires_grad=True)

    y_tor = a_tor @ b_tor - c_tor**2
    y_tor.backward(torch.ones_like(y_tor))
    
    assert np.allclose(a.grad, a_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(b.grad, b_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(c.grad, c_tor.grad.detach().numpy(), atol=1e-6)
    
    
def test_grad_broadcast():
    ###### NNCL ########
    a = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    b = Tensor(shape=(1, 5), dtype=Tensor.FLOAT32, grad_en=True).rand(-1, 1)
    
    y = a + b
    y.backward()
            
    ######### TORCH ###########    
    a_tor = torch.tensor(a.data, dtype=torch.float32, requires_grad=True)
    b_tor = torch.tensor(b.data, dtype=torch.float32, requires_grad=True)
    
    y_tor = a_tor + b_tor
    y_tor.backward(torch.ones_like(y_tor))    
        
    assert np.allclose(a.grad, a_tor.grad.detach().numpy(), atol=1e-6)
    assert np.allclose(b.grad, b_tor.grad.detach().numpy(), atol=1e-6)
    
    
    
    
