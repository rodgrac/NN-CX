from nncx.tensor import Tensor


class MaxPool2d:
    def __init__(self, backend, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.backend = backend
        
    def __call__(self, x: Tensor):
        return self.forward_opt(x)
    
    
    def forward_opt(self, x: Tensor):
        B, C, H, W = x.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        out = Tensor(shape=(B, C, H_out, W_out), backend=self.backend, grad_en=x.grad_en)
        
        # im2col
        cols = []
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                patch = x.data[:, :, i : i + self.kernel_size, j : j + self.kernel_size]     # (B, Cin, K, K)
                cols.append(patch.reshape(B, C, -1))       # (B, Cin, K*K)
        cols = self.backend.stack(cols, axis=-1)        # (B, Cin, K*K, Hout*Wout)
        
        out = self.backend.max(cols, axis=2).reshape(B, C, H_out, W_out)
        out = Tensor(out, backend=x.backend, grad_en=x.grad_en)
                
        if out.grad_en:            
            def _backward(grad):
                max_idx = self.backend.argmax(cols, axis=2).reshape(B*C, -1)

                dx_cols = self.backend.zeros((B * C, cols.shape[-1], cols.shape[-2]), x.dtype)  # (B*Cin, Hout*Wout, K*K)
                
                # Scatter
                dx_cols[self.backend.arange(dx_cols.shape[0])[:, None],
                        self.backend.arange(dx_cols.shape[1]),
                        max_idx] = grad.reshape(B*C, -1)    
                
                dx_cols = dx_cols.transpose(0, 2, 1).reshape(B, C, self.kernel_size, self.kernel_size, -1)    # (B, Cin, K, K, Hout*Wout)
                
                # col2im
                dx = self.backend.zeros(x.shape, x.dtype)
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
        
        out = Tensor(shape=(B, C, H_out, W_out), backend=self.backend, grad_en=x.grad_en)
        
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
                dx = self.backend.zeros(x.data.shape, x.dtype)
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