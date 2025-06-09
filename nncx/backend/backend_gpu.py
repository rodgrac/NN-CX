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
    
    def rand(self, shape, dtype, min_val, max_val):
        if dtype == DataType.INT32:
            return cp.random.randint(min_val, max_val + 1, shape, dtype=cp.int32)
        elif dtype == DataType.FLOAT32:
            return cp.random.uniform(min_val, max_val, shape).astype(cp.float32)
                
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
    
    def sum(self, x, axis=None, keepdims=False):
        return cp.sum(x, axis=axis, keepdims=keepdims)
    
    def add(self, a, b):
        return cp.add(a, b)
    
    def mul(self, a, b):
        return cp.mul(a, b)
    
    def exp(self, x):
        return cp.exp(x)
    
    def log(self, x):
        return cp.log(x)
    
    def argmax(self, x, axis=-1):
        return cp.argmax(x, axis=axis)
    
    def max(self, x, axis=None, keepdims=False):
        return cp.max(x, axis=axis, keepdims=keepdims)
    
    def diagflat(self, x):
        return cp.diagflat(x)
    
    def pad(self, x, pad_width, mode="constant"):
        return cp.pad(x, pad_width, mode)
    
    def __repr__(self):
        return BackendType.GPU