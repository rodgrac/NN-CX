import copy
import uuid
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
    
    _no_grad = False        # static var to control grad compute
    
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
        self.size = self.data.size
        self.ndims = self._ndims()
        
        self.uuid = uuid.uuid4()
        self.grad_en = grad_en and not self._no_grad
        self.grad_fn = None
        self._prev = [] if self.grad_en else None
        
        self._zero_grad()
        
    def __hash__(self):
        return hash(self.uuid)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return np.array_equal(self.data, other.data)
        return False
    
    def _zero_grad(self):
        if self.grad_en:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None
        
    def _ndims(self):
        ndims = 0
        for dim_val in self.shape:
            if dim_val > 1:
                ndims += 1
                
        return ndims
    
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
                if input_t is None:         # grad_en=false node
                    continue
                                
                if input_t.grad.shape != grads[i].shape:
                    reduce_axes = Tensor._get_grad_reduce_axes(input_t, grads[i])
                    keepdims = True if len(grads[i].shape) == len(input_t.shape) else False
                    input_t.grad += np.sum(grads[i], axis=reduce_axes, keepdims=keepdims)
                else:
                    input_t.grad += grads[i]
    
    @staticmethod
    def _get_grad_reduce_axes(tensor, grad_acc):       # Return axes to collapse update gradient to match tensor
        axes = []
        dims_ex = len(grad_acc.shape) - len(tensor.shape)
        axes.extend(np.arange(dims_ex))
        
        for ax, (val1, val2) in enumerate(zip(tensor.shape, grad_acc.shape[dims_ex:])):
            if val1 != val2:
                assert val1 == 1, "Shape mismatch for reduced gradient update"
                axes.append(ax + dims_ex)
        return tuple(axes)
            
                
    def detach(self):
        return Tensor(self.data, grad_en=False)
        
                
    def to_dtype(self, dtype):
        self.data = self.data.astype(self.dtype_map[dtype]['np'])
        self.dtype = dtype
        return self
        
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)

        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        
        out = Tensor(self.data + other.data, grad_en=self.grad_en or other.grad_en)
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad, grad]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)
        
        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        
        out = Tensor(self.data * other.data, grad_en=self.grad_en or other.grad_en)
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only (int, float) are supported exponent types"
        
        out = Tensor(self.data ** other, grad_en=self.grad_en)
        
        out.grad_en = self.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other * self.data ** (other - 1)]
            out._prev = [self]
        
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype)
        
        if self.ndims == 1 and other.ndims == 1:    # 1-d 
            out = Tensor(np.dot(self.data, other.data), grad_en=self.grad_en or other.grad_en)
        elif other.ndims == 0:                      # Scalar
            out = self * other
        else:
            out = Tensor(np.matmul(self.data, other.data), grad_en=self.grad_en or other.grad_en)    
            
        if out.grad_en:
            if self.ndims == 1 and other.ndims == 1:
                out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
            elif other.ndims == 0:                  # Handled  by __mul__
                pass
            else:
                out.grad_fn = lambda grad: [np.matmul(grad, np.moveaxis(other.data, -1, -2)), np.matmul(np.moveaxis(self.data, -1, -2), grad)]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
            
        return out
    
    
    def exp(self):
        out = Tensor(np.exp(self.data), grad_en=self.grad_en)
        
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
    
    
    @staticmethod
    def stack(tensors, axis=0):
        out_data = [t.data for t in tensors]
        out = Tensor(np.stack(out_data, axis=axis), grad_en=np.any([t.grad_en for t in tensors]))
        
        if out.grad_en:
            out.grad_fn = lambda grad: [np.expand_dims(grad, axis=axis) for t in tensors]
            for t in tensors:
                out._prev.append(t if t.grad_en else None)
            
        return out
            
            
    @staticmethod
    def no_grad(condition=True):
        class _NoGradContext:
            def __enter__(self):
                if condition:
                    Tensor._no_grad = True
            
            def __exit__(self, exc_type, exc_value, traceback):
                if condition:
                    Tensor._no_grad = False
        
        return _NoGradContext()
    
            
    def __repr__(self) -> str:
        return f"Tensor(data=\n{self.data}, shape={self.shape}, dtype={self.dtype_map[self.dtype]['str']}, grad_en={self.grad_en})"
        