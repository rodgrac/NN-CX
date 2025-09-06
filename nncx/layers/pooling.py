from nncx.tensor import Tensor


class MaxPool2d:
    def __init__(self, backend, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.backend = backend
        
    def __call__(self, x: Tensor):
        return self.forward(x)
    
    def forward(self, x: Tensor):
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