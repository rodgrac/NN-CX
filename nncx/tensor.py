import copy
import uuid
import numpy as np
import cupy as cp
from typing import Tuple

from nncx import utils
from nncx.enums import DataType, BackendType

class Tensor:
    
    _no_grad = False        # static var to control grad compute
    
    def __init__(self, data=None, shape=None, dtype=DataType.FLOAT32, backend_type=BackendType.CPU, grad_en=False) -> None:
        self.backend_type = backend_type
        
        self.dtype_map = {
            DataType.FLOAT32 : self.backend.float32,
            DataType.INT32 : self.backend.int32
        }
        
        if data is not None:
            if np.isscalar(data):       # FIXME: don't convert to 2d tensor, keep it 1d
                data = [[data]]     
            
            self.data = self.backend.asarray(data, order='C', dtype=self.dtype_map[dtype])    
            
        elif shape is not None:
            assert isinstance(shape, tuple), "shape argument needs to be a tuple!"
                
            self.data = self.backend.zeros(shape, self.dtype_map[dtype])
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
        
    
    @property
    def backend(self):
        return _get_backend_obj(self.backend_type)
        
    def __hash__(self):
        return hash(self.uuid)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return self.backend.array_equal(self.data, other.data)
        return False
    
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx], dtype=self.dtype, backend_type=self.backend_type, grad_en=self.grad_en)
    
    
    def _zero_grad(self):
        if self.grad_en:
            self.grad = self.backend.zeros(self.data.shape, self.dtype_map[self.dtype])
        else:
            self.grad = None
        
    def _ndims(self):
        ndims = 0
        for dim_val in self.shape:
            ndims += 1
            
        return ndims
    
    def _update_attrs(self):
        self.shape = self.data.shape
        self.size = self.data.size
        self.ndims = self._ndims()
    
    def backward(self, grad=None):
        if grad is None:  
            grad = self.backend.full(self.data.shape, 1.0, self.dtype_map[self.dtype])       
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
                    input_t.grad += self.backend.sum(grads[i], axis=reduce_axes, keepdims=keepdims)
                else:
                    input_t.grad += grads[i]
    
    @staticmethod
    def _get_grad_reduce_axes(tensor, grad_acc):       # Return axes to collapse update gradient to match tensor
        axes = []
        dims_ex = len(grad_acc.shape) - len(tensor.shape)
        axes.extend(np.arange(dims_ex))
        
        for ax, (val1, val2) in enumerate(zip(tensor.shape, grad_acc.shape[dims_ex:])):
            if val1 != val2:
                assert val1 == 1, f"Shape mismatch for reduced gradient update => ({tensor.shape}), ({grad_acc.shape[dims_ex:]})"
                axes.append(ax + dims_ex)
        return tuple(axes)
            
                
    def detach(self):
        return Tensor(self.data, dtype=self.dtype, backend_type=self.backend_type, grad_en=False)
    
    
    def to(self, backend_type):
        if self.backend_type == backend_type:
            return self
        
        return Tensor(self.data, dtype=self.dtype, backend_type=backend_type, grad_en=self.grad_en)
        
                
    def to_dtype(self, dtype):
        self.data = self.data.astype(self.dtype_map[dtype])
        self.dtype = dtype
        return self
    
    def get(self):
        if self.backend_type == BackendType.GPU:
            return self.data.get()
        else:
            return self.data
        
    def reshape(self, shape: Tuple):
        out = Tensor(self.data.reshape(*shape),
                     dtype=self.dtype, 
                     backend_type=self.backend_type,
                     grad_en=self.grad_en
                    )

        if out.grad_en:
            out.grad_fn = lambda grad: [grad.reshape(self.data.shape)]
            out._prev = [self]
        
        return out
    
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype, backend_type=self.backend_type)

        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        assert type(self.backend) == type(other.backend), "Tensors need to be of same backend type!"
        
        out = Tensor(self.backend.add(self.data, other.data), 
                     dtype=self.dtype, 
                     backend_type=self.backend_type,
                     grad_en=self.grad_en or other.grad_en
                     )
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad, grad]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype, backend_type=self.backend_type)
        
        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        assert type(self.backend) == type(other.backend), "Tensors need to be of same backend type!"
        
        out = Tensor(self.backend.multiply(self.data, other.data), 
                     dtype=self.dtype,
                     backend_type=self.backend_type,
                     grad_en=self.grad_en or other.grad_en
                     )
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only (int, float) are supported exponent types"
        assert type(self.backend) == type(other.backend), "Tensors need to be of same backend type!"
        
        out = Tensor(self.data ** other, dtype=self.dtype, backend_type=self.backend_type, grad_en=self.grad_en)
        
        out.grad_en = self.grad_en
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * other * self.data ** (other - 1)]
            out._prev = [self]
        
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other, dtype=self.dtype, backend_type=self.backend_type)
            
        assert self.dtype == other.dtype, "Tensors need to be of same data type!"
        assert type(self.backend) == type(other.backend), "Tensors need to be of same backend type!"
        
        if self.ndims == 1 and other.ndims == 1:    # 1-d 
            out = Tensor(self.backend.dot(self.data, other.data), 
                         dtype=self.dtype,
                         backend_type=self.backend_type,
                         grad_en=self.grad_en or other.grad_en
                         )
            
        elif other.ndims == 0:                      # Scalar
            out = self * other
            
        else:
            out = Tensor(self.backend.matmul(self.data, other.data), 
                         dtype=self.dtype,
                         backend_type=self.backend_type,
                         grad_en=self.grad_en or other.grad_en
                         )    
            
        if out.grad_en:
            if self.ndims == 1 and other.ndims == 1:
                out.grad_fn = lambda grad: [grad * other.data, grad * self.data]
            elif other.ndims == 0:                  # Handled  by __mul__
                pass
            else:
                out.grad_fn = lambda grad: [self.backend.matmul(grad, self.backend.moveaxis(other.data, -1, -2)), self.backend.matmul(self.backend.moveaxis(self.data, -1, -2), grad)]
            out._prev = [self if self.grad_en else None, other if other.grad_en else None]
            
        return out
    
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), 
                     dtype=self.dtype,
                     backend_type=self.backend_type, 
                     grad_en=self.grad_en
                    )
                        
        if out.grad_en:
            if not len(axes):
                axes = tuple(reversed(range(out.ndims)))
            axes = np.argsort(axes).tolist()
            out.grad_fn = lambda grad: [grad.transpose(axes)]
            out._prev = [self]
            
        return out
    
    @property
    def T(self):
        return self.transpose()
    
    def exp(self):
        out = Tensor(self.backend.exp(self.data), 
                     dtype=self.dtype,
                     backend_type=self.backend_type, 
                     grad_en=self.grad_en
                     )
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * out.data]
            out._prev = [self]
        
        return out
    
    
    def argmax(self, axis=None):
        out = Tensor(self.backend.argmax(self.data, axis=axis), 
                     dtype=self.dtype,
                     backend_type=self.backend_type, 
                     grad_en=False
                     )
        
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
        if self.dtype == DataType.INT32:
            self.data[:] = self.backend.random.randint(min_val, max_val + 1, self.shape, dtype=self.dtype_map[self.dtype])
        elif self.dtype == DataType.FLOAT32:
            self.data[:] = self.backend.random.uniform(min_val, max_val, self.shape).astype(self.dtype_map[self.dtype])
                        
        return self
    
    def randn(self):
        assert self.dtype == DataType.FLOAT32, "Data type needs to be float"
       
        self.data[:] = self.backend.random.randn(*self.shape)
        
        return self
    
    def clone(self):
        return copy.deepcopy(self)
    
    
    @staticmethod
    def stack(tensors, axis=0):
        out_data = [t.data for t in tensors]
        out = Tensor(tensors[0].backend.stack(out_data, axis=axis), 
                     backend_type=tensors[0].backend_type, 
                     dtype=tensors[0].dtype,
                     grad_en=np.any([t.grad_en for t in tensors]))
        
        if out.grad_en:
            out.grad_fn = lambda grad: [tensors[0].backend.expand_dims(grad, axis=axis) for t in tensors]
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
        return f"Tensor(data=\n{self.data}, shape={self.shape}, dtype={self.dtype}, backend={self.backend_type}, grad_en={self.grad_en})"
        
        
def _get_backend_obj(backend_type):
    if backend_type == BackendType.CPU:
        return np
    elif backend_type == BackendType.GPU:
        return cp
    else:
        raise NotImplementedError("Backend not supported!")