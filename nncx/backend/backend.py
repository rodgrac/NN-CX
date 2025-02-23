from abc import ABC, abstractmethod


class Backend(ABC):
    def __init__(self):
        super().__init__()
        
    def array(self, data):
        raise NotImplementedError
    
    def zeros(self, shape, dtype):
        raise NotImplementedError 
    
    def full(self, shape, value, dtype):
        raise NotImplementedError 
    
    def rand(self, data, min_val, max_val):
        raise NotImplementedError 
    
    def randn(self, shape):
        raise NotImplementedError 
    
    def array_equal(self, a, b):
        raise NotImplementedError 
    
    def moveaxis(self, x, src_axis, dest_axis):
        raise NotImplementedError 
    
    def expand_dims(self, x, axis):
        raise NotImplementedError 
    
    def stack(self, tensors, axis):
        raise NotImplementedError 
    
    def dot(self, a, b):
        raise NotImplementedError 
    
    def matmul(self, a, b):
        raise NotImplementedError
    
    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError 
    
    def add(self, a, b):
        raise NotImplementedError    
    
    def mul(self, a, b):
        raise NotImplementedError 
        
    def exp(self, x):
        raise NotImplementedError 
    
    def log(self, x):
        raise NotImplementedError 
    
    def argmax(self, x, axis=-1):
        raise NotImplementedError
    
    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError
    
    def diagflat(self, x):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError 
    

