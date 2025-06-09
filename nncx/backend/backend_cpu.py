import numpy as np

from nncx.backend.backend import Backend
from nncx.enums import DataType, BackendType


class CPUBackend(Backend):
    def __init__(self):
        super().__init__()
        
        self.dtype_map = {
            DataType.FLOAT32 : np.float32,
            DataType.INT32 : np.int32
        }
    
    def array(self, data, dtype):
        return np.asarray(data, dtype=self.dtype_map[dtype])
    
    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=self.dtype_map[dtype])
    
    def full(self, shape, value, dtype):
        return np.full(shape, value, dtype=dtype)
    
    def rand(self, shape, dtype, min_val, max_val):
        if dtype == DataType.INT32:
            return np.random.randint(min_val, max_val + 1, shape, dtype=np.int32)
        elif dtype == DataType.FLOAT32:
            return np.random.uniform(min_val, max_val, shape).astype(np.float32)
            
    
    def randn(self, shape):
        return np.random.randn(*shape)
    
    def array_equal(self, a, b):
        return np.array_equal(a.data, b.data)
    
    def moveaxis(self, x, src_axis, dest_axis):
        return np.moveaxis(x, src_axis, dest_axis)
    
    def expand_dims(self, x, axis):
        return np.expand_dims(x, axis=axis)
    
    def stack(self, tensors, axis):
        return np.stack(tensors, axis=axis)
    
    def dot(self, a, b):
        return np.dot(a, b)
    
    def matmul(self, a, b):
        return np.matmul(a, b)
    
    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def add(self, a, b):
        return np.add(a, b)
    
    def mul(self, a, b):
        return np.multiply(a, b)
        
    def exp(self, x):
        return np.exp(x)
    
    def log(self, x):
        return np.log(x)
    
    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)
    
    def argmax(self, x, axis=None):
        return np.argmax(x, axis=axis)
    
    def diagflat(self, x):
        return np.diagflat(x)
    
    def pad(self, x, pad_width, mode="constant"):
        return np.pad(x, pad_width, mode)
    
    def __repr__(self):
        return BackendType.CPU
        
    