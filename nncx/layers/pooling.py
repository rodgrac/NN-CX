from nncx.tensor import Tensor, _get_backend_obj


class MaxPool2d:
    def __init__(self, backend_type, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.backend_type = backend_type
        
    @property
    def backend(self):
        return _get_backend_obj(self.backend_type)
        
    def __call__(self, x: Tensor):
        return self.forward_opt2(x)

    
    def forward_opt2(self, x: Tensor):
        B, C, H, W = x.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        # (B, C, H-K+1, W-K+1, K, K)
        windows = self.backend.lib.stride_tricks.sliding_window_view(
            x.data, (self.kernel_size, self.kernel_size), axis=(-2, -1))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        out = windows.max(axis=(-1, -2))
        out = Tensor(out, backend_type=x.backend_type, grad_en=x.grad_en)
        
        max_mask = (windows == out.data[..., None, None])
               
        if out.grad_en:            
            def _backward(grad):
                # (B, C, Hout, Wout)
                max_idx = self.backend.argmax(max_mask.reshape(B, C, H_out, W_out, -1), axis=-1)
                
                dh, dw = max_idx // self.kernel_size, max_idx % self.kernel_size
                
                # Compute absolute coordinates
                h_base = self.backend.arange(H_out)[None, None, :, None] * self.stride
                w_base = self.backend.arange(W_out)[None, None, None, :] * self.stride
                h_abs = h_base + dh
                w_abs = w_base + dw
                
                batch_ch_idx = self.backend.repeat(self.backend.arange(B*C), H_out*W_out) 

                dx = self.backend.zeros_like(x.data, dtype=x.dtype_map[x.dtype])  
                
                self.backend.add.at(dx.reshape(B * C, H, W),
                                    (batch_ch_idx, h_abs.reshape(-1), w_abs.reshape(-1)),
                                    grad.reshape(-1))
                
                dx = dx.reshape(B, C, H, W)
                        
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    def forward_opt1(self, x: Tensor):
        B, C, H, W = x.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        out = Tensor(shape=(B, C, H_out, W_out), backend_type=x.backend_type, grad_en=x.grad_en)
        
        # im2col
        cols = []
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                patch = x.data[:, :, i : i + self.kernel_size, j : j + self.kernel_size]     # (B, Cin, K, K)
                cols.append(patch.reshape(B, C, -1))       # (B, Cin, K*K)
        cols = self.backend.stack(cols, axis=-1)        # (B, Cin, K*K, Hout*Wout)
        
        out = self.backend.max(cols, axis=2).reshape(B, C, H_out, W_out)
        out = Tensor(out, backend_type=x.backend_type, grad_en=x.grad_en)
                
        if out.grad_en:            
            def _backward(grad):
                max_idx = self.backend.argmax(cols, axis=2).reshape(B*C, -1)

                dx_cols = self.backend.zeros((B * C, cols.shape[-1], cols.shape[-2]), x.dtype_map[x.dtype])  # (B*Cin, Hout*Wout, K*K)
                
                # Scatter
                dx_cols[self.backend.arange(dx_cols.shape[0])[:, None],
                        self.backend.arange(dx_cols.shape[1]),
                        max_idx] = grad.reshape(B*C, -1)    
                
                dx_cols = dx_cols.transpose(0, 2, 1).reshape(B, C, self.kernel_size, self.kernel_size, -1)    # (B, Cin, K, K, Hout*Wout)
                
                # col2im
                dx = self.backend.zeros(x.shape, x.dtype_map[x.dtype])
                idx = 0
                for i in range(0, H_out*self.stride, self.stride):
                    for j in range(0, W_out*self.stride, self.stride):
                        dx[:, :, i: i+self.kernel_size, j: j+self.kernel_size] += dx_cols[:, :, :, :, idx]
                        idx += 1
                        
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    
    def forward_basic(self, x: Tensor):
        B, C, H, W = x.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        out = Tensor(shape=(B, C, H_out, W_out), backend_type=x.backend_type, grad_en=x.grad_en)
        
        cache = {}
        for row in range(H_out):
            for col in range(W_out):
                row_s = row * self.stride
                col_s = col * self.stride
                
                region = x.data[:, :, row_s : row_s + self.kernel_size,
                                col_s : col_s + self.kernel_size]
                max_val = self.backend.max(region, axis=(2, 3))
                out.data[:, :, row, col] = max_val
                
                mask = (region == self.backend.expand_dims(max_val, axis=(-1, -2))) 
                cache[(row, col)] = mask
                
        if out.grad_en:
            def _backward(grad):
                dx = self.backend.zeros(x.data.shape, x.dtype_map[x.dtype])
                for row in range(H_out):
                    for col in range(W_out):
                        row_s = row * self.stride
                        col_s = col * self.stride

                        dx[:, :, row_s : row_s + self.kernel_size,
                           col_s : col_s + self.kernel_size] += self.backend.expand_dims(grad[:, :, row, col], axis=(-1, -2)) * cache[(row, col)]
        
                return [dx]
            
            out.grad_fn = _backward
            out._prev = [x]
        
        return out
    
    
    def __repr__(self):
        return f"MaxPool2d"