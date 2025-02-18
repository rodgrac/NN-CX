from typing import Any
from nncl.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, backend, bias=True) -> None:
        self.w = Tensor(shape=(in_features, out_features), backend=backend, grad_en=True).randn()
        if bias:
            self.bias = Tensor(shape=(1, out_features), backend=backend, grad_en=True)
        else:
            self.bias = None
            
        self._params = [self.w, self.bias] if self.bias is not None else [self.w]
        
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x):
        x = x @ self.w
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def __repr__(self) -> str:
        return f"Linear({self.w.shape}, bias={self.bias!=None})"
    