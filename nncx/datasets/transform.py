import numpy as np

class Transform:
    def __call__(self, tensor):
        raise NotImplementedError
    
class Standardize(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std
    
    def invert(self, tensor):
        return tensor * self.std + self.mean

class Normalize(Transform):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)
    
    def invert(self, tensor):
        return tensor * (self.max_val - self.min_val) + self.min_val
    

class Flatten(Transform):
    def __call__(self, tensor):
        return tensor.flatten()
    
    
class OneHotEncode(Transform):
    def __init__(self, num_classes):
        self.enc = np.eye(num_classes)
        
    def __call__(self, tensor):
        return self.enc[np.array(tensor)]
