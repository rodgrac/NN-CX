import copy
import numpy as np

from nncl import utils

class Tensor:
    # Enums
    FLOAT32 = 0
    INT32 = 1
    
    dtype_map = {
        FLOAT32 : {'str': 'float32', 'np': np.float32},
        INT32 : {'str': 'int32', 'np': np.int32}
    }
    
    def __init__(self, data=None, shape=None, dtype=FLOAT32, grad_en=False) -> None:
        if data is not None:
            if np.isscalar(data):
                data = [[data]]
            self.data = np.asarray(data, dtype=self.dtype_map[dtype]['np'])
        elif shape is not None:
            assert isinstance(shape, tuple), "shape argument needs to be a tuple!"
                
            self.data = np.zeros(shape, dtype=self.dtype_map[dtype]['np'])
        else:
            raise ValueError("Either data or shape needs to be specified!")
        
        self.dtype = dtype
        self.shape = self.data.shape
        self.ndims = self._ndims()
        self.grad_en = grad_en
        self.grad_fn = None
        self._prev = []
        
        self._zero_grad()
        
    def __hash__(self):
        return hash(self.data.tobytes())
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return np.array_equal(self.data, other.data)
        return False
    
    def _zero_grad(self):
        self.grad = np.zeros_like(self.data)
        
    def _ndims(self):
        ndims = 0
        for dim_val in self.shape:
            if dim_val > 1:
                ndims += 1
    
    def backward(self, grad=None):
        if grad is None:         
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad
        
        sorted_tensors = utils.topo_sort(self)
        
        for tensor in sorted_tensors[::-1]:
            tensor._backward()
            
    def _backward(self):
        if self.grad_fn is not None:
            grads = self.grad_fn(self.grad)
            for i, input_t in enumerate(self._prev):
                if input_t.grad.shape != grads[i].shape:
                    input_t.grad = np.full_like(grads[i], input_t.grad)
                input_t.grad += grads[i]
        
                
    def to_dtype(self, dtype):
        self.data = self.data.astype(self.dtype_map[dtype]['np'])
        self.dtype = dtype
        return self
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)

        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        
        out = Tensor(self.data + other.data)
        
        out.grad_en = self.grad_en or other.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad, grad]
        out._prev = [self, other]
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)
        
        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        
        out = Tensor(self.data * other.data)
        
        out.grad_en = self.grad_en or other.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
        out._prev = [self, other]
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only (int, float) are supported exponent types"
        
        out = Tensor(self.data ** other)
        
        out.grad_en = self.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other * self.data ** (other - 1)]
        out._prev = [self]
        
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)
        
        if self.ndims == 1 and other.ndims == 1:    # 1-d 
            out = Tensor(np.dot(self.data, other.data))
        elif other.ndims == 0:                      # Scalar
            out = Tensor(self.data * other.data)
        else:
            out = Tensor(np.matmul(self.data, other.data))    
            
        out.grad_en = self.grad_en or other.grad_en
        if out.grad_en:
            if self.ndims == 1 and other.ndims == 1:
                out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
            elif other.ndims == 0:                  # Handled  by __mul__
                pass
            else:
                out.grad_fn = lambda grad: [np.matmul(grad, other.data.T), np.matmul(self.data.T, grad)]
        out._prev = [self, other]
            
        return out
    
    
    def exp(self):
        out = Tensor(np.exp(self.data))
        
        out.grad_en = self.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * out.data]
        out._prev = [self]
        
        return out

    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def rand(self, min_val, max_val):
        if self.dtype == self.INT32:
            self.data[:] = np.random.randint(min_val, max_val + 1, self.shape, dtype=np.int32)
        elif self.dtype == self.FLOAT32:
            self.data[:] = np.random.uniform(min_val, max_val, self.shape).astype(np.float32)
            
        return self
    
    def randn(self):
        assert self.dtype == self.FLOAT32, "Data type needs to be float"
       
        self.data[:] = np.random.randn(*self.shape)
        
        return self
    
    def clone(self):
        return copy.deepcopy(self)
    
            
    def __repr__(self) -> str:
        return f"Tensor(data=\n{self.data}, dtype={self.dtype_map[self.dtype]['str']}, grad_en={self.grad_en})"
        