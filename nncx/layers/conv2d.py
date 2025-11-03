from typing import Any
from nncx.tensor import Tensor, _get_backend_obj

class Conv2d:
    def __init__(self, in_channels, out_channels, backend_type, kernel_size=3, stride=1, pad=False,
                 bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        self.backend_type = backend_type
        
        self.w = Tensor(shape=(out_channels, in_channels, kernel_size, kernel_size),
                        backend_type=backend_type, grad_en=True).randn()
        self.w.data *= (2.0 / (in_channels * kernel_size**2))**0.5
        
        if bias:
            self.bias = Tensor(shape=(out_channels,), backend_type=backend_type, grad_en=True)
        else:
            self.bias = None
            
        self._params = [self.w, self.bias] if self.bias is not None else [self.w]
        
    
    @property
    def backend(self):
        return _get_backend_obj(self.backend_type)
        
        
    def __call__(self, x: Tensor):
        return self.forward_opt2(x)
    
    def _im2col_idxs(self, Cin, H, W, K, pad, stride):
        # Compute output spatial dims
        Hout = (H - K + 2 * pad) // stride + 1
        Wout = (W - K + 2 * pad) // stride + 1
    
        # KxK window
        i0 = self.backend.repeat(self.backend.arange(K), K)     # (K*K)
        i0 = self.backend.tile(i0, Cin)                         # (C*K*K)
        j0 = self.backend.tile(self.backend.arange(K), K * Cin)
        
        # Channel
        k_idx = self.backend.repeat(self.backend.arange(Cin), K * K).reshape(-1, 1)        # (C*K*K)
        
        # Patches
        i1 = self.backend.repeat(self.backend.arange(Hout), Wout) * stride          # (Hout*Wout)
        j1 = self.backend.tile(self.backend.arange(Wout), Hout) * stride            # (Hout*Wout)
        
        # Broadcast to get all (C*K*K, H_out*W_out)
        i = i0[:, None] + i1[None, :]
        j = j0[:, None] + j1[None, :]
        
        return k_idx, i, j
        
    
    def forward_opt2(self, x: Tensor):
        B, C_in, H, W = x.shape
        
        if self.pad:
            pad_w = self.kernel_size // 2
        else:
            pad_w = 0
            
        H_out = (H - self.kernel_size + 2 * pad_w) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * pad_w) // self.stride + 1
                
        x_pad = self.backend.pad(x.data, ((0, 0), (0, 0), (pad_w, pad_w), (pad_w, pad_w)))
        
        # im2col
        cache_key = (C_in, H, W, self.kernel_size, pad_w, self.stride)
        if not hasattr(self, "_im2col_cache"):
            self._im2col_cache = {}
        if cache_key not in self._im2col_cache:
            k_idx, i_idx, j_idx = self._im2col_idxs(C_in, H, W, self.kernel_size, pad_w, self.stride)
            self._im2col_cache[cache_key] = (k_idx, i_idx, j_idx)
        else:
            k_idx, i_idx, j_idx = self._im2col_cache[cache_key]
            
        cols = x_pad[:, k_idx, i_idx, j_idx]    # (B, Cin*K*K, Hout*Wout)
        
        # flatten weights
        w_col = self.w.data.reshape(self.out_channels, -1)  # (Cout, Cin*K*K)
        
        # GEMM
        out = (cols.transpose(0, 2, 1) @ w_col.T).transpose(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)    # (B, Cout, Hout, Wout)
        
        if self.bias:
            out += self.bias.data[None, :, None, None]
            
        out = Tensor(out, backend_type=x.backend_type, grad_en=x.grad_en)
        

        if out.grad_en:
            def _backward(grad):
                grad_col = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)            # (B*Hout*Wout, Cout)
                cols_2d = cols.transpose(0, 2, 1).reshape(grad_col.shape[0], -1)                   # (B*Hout*Wout, Cin*K*K)
                   
                # bias grad
                if self.bias is not None:
                    self.bias.grad = self.backend.sum(grad, axis=(0, 2, 3))
                    
                self.w.grad = (grad_col.T @ cols_2d).reshape(self.w.data.shape)                    # (Cout, Cin, K, K)
                
                dx_cols = (grad_col @ w_col).reshape(B, -1, cols_2d.shape[-1]).transpose(0, 2, 1)     # (B, Cin*K*K, Hout*Wout)
                
                #col2im - Scatter add
                # add every dx_cols[:, p, q] into the corresponding dx[:, k_idx[p], i_idx[p, q], j_idx[p, q]]
                dx = self.backend.zeros(x_pad.shape, dtype=x.dtype_map[x.dtype])
                self.backend.add.at(dx, (slice(None), k_idx, i_idx, j_idx), dx_cols)
                
                if self.pad:
                    dx = dx[:, :, pad_w:-pad_w, pad_w:-pad_w]
                
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    
    def forward_opt1(self, x: Tensor):
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
        out = self.backend.einsum("bpc,oc->bop", cols.transpose(0, 2, 1), w_col).reshape(B, self.out_channels, H_out, W_out)
        
        if self.bias:
            out += self.bias.data[None, :, None, None]
            
        out = Tensor(out, backend_type=x.backend_type, grad_en=x.grad_en)
        
                        
        if out.grad_en:
            def _backward(grad):
                grad_col = grad.reshape(B, self.out_channels, -1)
                                
                # bias grad
                if self.bias is not None:
                    self.bias.grad = self.backend.sum(grad, axis=(0, 2, 3))
                    
                # weight grad
                self.w.grad = self.backend.einsum("bop,bcp->oc", grad_col, cols).reshape(self.w.data.shape)
                
                # input grad
                dx_cols = self.backend.einsum("bop,ow->bwp", grad_col, w_col).reshape(B, C_in, self.kernel_size, self.kernel_size, -1)
                
                #col2im
                dx = self.backend.zeros(x_pad.shape, x.dtype_map[x.dtype])
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
        
        out = Tensor(shape=(B, self.out_channels, H_out, W_out), backend_type=x.backend_type, grad_en=x.grad_en)
        
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
                dx = self.backend.zeros(x_pad.shape, x.dtype_map[x.dtype])
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