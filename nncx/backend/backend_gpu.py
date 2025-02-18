import cupy as cp

from nncx.backend.backend import Backend
from nncx.enums import DataType, BackendType

class GPUBackend(Backend):
    def __init__(self):
        super().__init__()
        
        print('CuPy version:', cp.__version__)
        print('CUDA version:', cp.cuda.runtime.runtimeGetVersion())
        
        self.dtype_map = {
            DataType.FLOAT32 : cp.float32,
            DataType.INT32 : cp.int32
        }
        
    def array(self, data, dtype):
        return cp.asarray(data, dtype=self.dtype_map[dtype])
    
    def zeros(self, shape, dtype):
        return cp.zeros(shape, dtype=self.dtype_map[dtype])
    
    def full(self, shape, value, dtype):
        return cp.full(shape, value, dtype=dtype)
    
    def rand(self, data, min_val, max_val):
        if data.dtype == cp.int32:
            data[:] = cp.random.randint(min_val, max_val + 1, data.shape, dtype=cp.int32)
        elif data.dtype == cp.float32:
            data[:] = cp.random.uniform(min_val, max_val, data.shape).astype(cp.float32)
            
        return data
    
    def randn(self, shape):
        return cp.random.randn(*shape)
    
    def array_equal(self, a, b):
        return cp.array_equal(a.data, b.data)
    
    def moveaxis(self, x, src_axis, dest_axis):
        return cp.moveaxis(x, src_axis, dest_axis)
    
    def expand_dims(self, x, axis):
        return cp.expand_dims(x, axis=axis)
    
    def stack(self, tensors, axis):
        return cp.stack(tensors, axis=axis)
    
    def dot(self, a, b):
        return cp.dot(a, b)
    
    def matmul(self, a, b):
        return cp.matmul(a, b)
    
    def sum(self, x, axis, keepdims):
        return cp.sum(x, axis=axis, keepdims=keepdims)
    
    def add(self, a, b):
        return cp.add(a, b)
    
    def mul(self, a, b):
        return cp.mul(a, b)
    
    def exp(self, x):
        return cp.exp(x)
    
    def __repr__(self):
        return BackendType.GPU