from typing import Any
from nncx.tensor import Tensor


class Conv2d:
    def __init__(self, in_channels, out_channels, backend, kernel_size=3, stride=1, pad=False,
                 bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        self.backend = backend
        
        self.w = Tensor(shape=(out_channels, in_channels, kernel_size, kernel_size),
                        backend=backend, grad_en=True).randn()
        self.w.data *= (2.0 / (in_channels * kernel_size**2))**0.5
        
        if bias:
            self.bias = Tensor(shape=(out_channels,), backend=backend, grad_en=True)
        else:
            self.bias = None
            
        self._params = [self.w, self.bias] if self.bias is not None else [self.w]
        
    def __call__(self, x: Tensor):
        return self.forward(x)
    
    def forward(self, x: Tensor):
        B, C_in, H, W = x.shape
        
        if self.pad:
            pad_w = self.kernel_size // 2
        else:
            pad_w = 0
            
        H_out = (H - self.kernel_size + 2 * pad_w) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * pad_w) // self.stride + 1
        
        out = Tensor(shape=(B, self.out_channels, H_out, W_out), backend=self.backend, grad_en=x.grad_en)
        
        x_pad = self.backend.pad(x.data, ((0, 0), (0, 0), (pad_w, pad_w), (pad_w, pad_w)))
        
        for b in range(B):
            for c in range(self.out_channels):
                for row in range(H_out):
                    for col in range(W_out):
                        row_s = row * self.stride
                        col_s = col * self.stride

                        out.data[b, c, row, col] = self.backend.sum(x_pad[b, :, row_s : row_s + self.kernel_size, 
                                                                           col_s : col_s + self.kernel_size] \
                                                    * self.w.data[c]) + (self.bias.data[c] if self.bias is not None else 0)
                        
        if out.grad_en:
            def _backward(grad):                
                # bias grad
                if self.bias is not None:
                    self.bias.grad.fill(0)
                    
                    for c in range(self.out_channels):
                        self.bias.grad[c] = self.backend.sum(grad[:, c, :, :])
                    
                # weight and input grad
                self.w.grad.fill(0)
                dx = self.backend.zeros(x_pad.shape, x.dtype)
                for b in range(B):
                    for c in range(self.out_channels):
                        for row in range(H_out):
                            for col in range(W_out):
                                row_s = row * self.stride
                                col_s = col * self.stride
                                region = x_pad[b, :, row_s : row_s + self.kernel_size, 
                                                col_s : col_s + self.kernel_size]
                                self.w.grad[c] += region * grad[b, c, row, col]
                                
                                dx[b, :, row_s : row_s + self.kernel_size, 
                                          col_s : col_s + self.kernel_size] += self.w.data[c] * grad[b, c, row, col]
                                
                if self.pad:
                    dx = dx[:, :, pad_w:-pad_w, pad_w:-pad_w]
                    
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    def __repr__(self):
        return f"Conv2d"