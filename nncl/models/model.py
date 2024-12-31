from abc import ABC, abstractmethod
from typing import Any

import numpy as np

    
class Model(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._modelparams = []
        self.training = None
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def __call__(self, x) -> Any:
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
        param_dict = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_params'):
                param_dict[attr_name] = attr._params
                
        np.savez(save_path, **param_dict)
        print(f"Model parameters saved as {save_path}")
        
    def load_parameters(self, load_path):
        params_dict = np.load(load_path, allow_pickle=True)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_params'):
                for i in range(len(attr._params)):
                    attr._params[i] = params_dict[attr_name][i]
        print(f"Model parameters loaded from {load_path}")
        
                
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        