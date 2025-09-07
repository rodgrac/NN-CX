import numpy as np

class Transform:
    def __call__(self, tensor):
        raise NotImplementedError
    
class Standardize(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        
    def __call__(self, tensor):
        mean = self.mean
        std = self.std
        if tensor.ndim == 3:
            mean = np.expand_dims(self.mean, axis=(-1, -2))
            std = np.expand_dims(self.std, axis=(-1, -2))
        
        return (tensor - mean) / std
    
    def invert(self, tensor):
        mean = self.mean
        std = self.std
        if tensor.ndim == 3:
            mean = np.expand_dims(self.mean, axis=(-1, -2))
            std = np.expand_dims(self.std, axis=(-1, -2))
        
        return tensor * std + mean

class Normalize(Transform):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, tensor):
        return (tensor.astype(np.float32) - self.min_val) / (self.max_val - self.min_val)
    
    def invert(self, tensor):
        return tensor.astype(np.float32) * (self.max_val - self.min_val) + self.min_val
    

class Flatten(Transform):
    def __call__(self, tensor):
        return tensor.flatten()
    
    
class OneHotEncode(Transform):
    def __init__(self, num_classes):
        self.enc = np.eye(num_classes, dtype=np.int32)
        
    def __call__(self, tensor):
        return self.enc[np.array(tensor)]
