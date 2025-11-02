from typing import Any
from nncx.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, backend_type, bias=True) -> None:
        self.w = Tensor(shape=(out_features, in_features), backend_type=backend_type, grad_en=True).randn()
        self.w.data *= (2.0 / in_features)**0.5
         
        if bias:
            self.bias = Tensor(shape=(out_features,), backend_type=backend_type, grad_en=True)
        else:
            self.bias = None
            
        self._params = [self.w, self.bias] if self.bias is not None else [self.w]
        
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x: Tensor):
        x = x @ self.w.T
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def __repr__(self) -> str:
        return f"Linear({self.w.shape}, bias={self.bias!=None})"
    