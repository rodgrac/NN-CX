from typing import Any
import numpy as np

from nncx.tensor import Tensor


class ReLU:
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x):
        out = Tensor(data=np.maximum(0, x.data), backend=x.backend, grad_en=x.grad_en)
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * (x.data > 0)]
            out._prev = [x]
        
        return out
    
    def __repr__(self) -> str:
        return f"ReLU"
        
        
class Sigmoid:
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x):
        out = Tensor(1/(1 + x.backend.exp(-x.data)), backend=x.backend, grad_en=x.grad_en)
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * out.data * (1 - out.data)]
            out._prev = [x]
        
        return out
    
    def __repr__(self) -> str:
        return f"Sigmoid"
    
    
class Tanh:
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x):
        out = Tensor((x.backend.exp(x.data) - x.backend.exp(-x.data)) /(x.backend.exp(x.data) + x.backend.exp(-x.data)), 
                     backend=x.backend, grad_en=x.grad_en)
        
        if out.grad_en:
            out.grad_fn = lambda grad: [grad * (1 - out.data**2.0)]
            out._prev = [x]
        
        return out
    
    def __repr__(self) -> str:
        return f"Tanh"