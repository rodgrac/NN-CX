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
        return self.forward_opt(x)
    
    
    def forward_opt(self, x: Tensor):
        B, C_in, H, W = x.shape
        
        if self.pad:
            pad_w = self.kernel_size // 2
        else:
            pad_w = 0
            
        H_out = (H - self.kernel_size + 2 * pad_w) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * pad_w) // self.stride + 1
                
        x_pad = self.backend.pad(x.data, ((0, 0), (0, 0), (pad_w, pad_w), (pad_w, pad_w)))
        
        # im2col
        cols = []
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                patch = x_pad[:, :, i : i + self.kernel_size, j : j + self.kernel_size]     # (B, Cin, K, K)
                cols.append(patch.reshape(B, -1))       # (B, Cin*K*K)
        cols = self.backend.stack(cols, axis=-1)        # (B, Cin*K*K, Hout*Wout)
        
        # flatten weights
        w_col = self.w.data.reshape(self.out_channels, -1)  # (Cout, Cin*K*K)
        
        # GEMM
        out = self.backend.einsum("bpc,oc->bop", [cols.transpose(0, 2, 1), w_col]).reshape(B, self.out_channels, H_out, W_out)
        
        if self.bias:
            out += self.bias.data[None, :, None, None]
            
        out = Tensor(out, backend=self.backend, grad_en=x.grad_en)
        
                        
        if out.grad_en:
            def _backward(grad):
                grad_col = grad.reshape(B, self.out_channels, -1)
                                
                # bias grad
                if self.bias is not None:
                    self.bias.grad = self.backend.sum(grad, axis=(0, 2, 3))
                    
                # weight grad
                self.w.grad = self.backend.einsum("bop,bcp->oc", [grad_col, cols]).reshape(self.w.data.shape)
                
                # input grad
                dx_cols = self.backend.einsum("bop,ow->bwp", [grad_col, w_col]).reshape(B, C_in, self.kernel_size, self.kernel_size, -1)
                
                #col2im
                dx = self.backend.zeros(x_pad.shape, x.dtype)
                idx = 0
                for i in range(0, H_out*self.stride, self.stride):
                    for j in range(0, W_out*self.stride, self.stride):
                        dx[:, :, i: i+self.kernel_size, j: j+self.kernel_size] += dx_cols[:, :, :, :, idx]
                        idx += 1
                
                if self.pad:
                    dx = dx[:, :, pad_w:-pad_w, pad_w:-pad_w]
                
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    
    def forward_basic(self, x: Tensor):
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