import os
from abc import ABC, abstractmethod
from typing import Any

from nncx.tensor import _get_backend_obj
    
class Model(ABC):
    def __init__(self, backend_type) -> None:
        super().__init__()
        self._modelparams = []
        self.training = None
        self.backend_type = backend_type
        
    @property
    def backend(self):
        return _get_backend_obj(self.backend_type)
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def __call__(self, x) -> Any:
        return self.forward(x)
    
    
    def predict(self, x):
        return self.forward(x)
    
    
    def parameters(self):
        self._modelparams.clear()
        self._collect_params()
        return self._modelparams
        
    def _collect_params(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_params'):
                self._modelparams.extend(attr._params)
        
    def save_parameters(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        param_dict = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_params'):
                param_dict[attr_name] = []
                for i, param in enumerate(attr._params):
                    param_dict[attr_name + '_p' + str(i)] = param.data
            
            self.backend.savez(save_path, **param_dict)
            
        print(f"Model parameters saved as {save_path}")
        
    def load_parameters(self, load_path):
        params_dict = self.backend.load(load_path, allow_pickle=True)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_params'):
                for i, param in enumerate(attr._params):
                    param.data = params_dict[attr_name + '_p' + str(i)]
        print(f"Model parameters loaded from {load_path}")
        
                
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        