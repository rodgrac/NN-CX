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
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        
        return (tensor - mean) / std
    
    def invert(self, tensor):
        mean = self.mean
        std = self.std
        if tensor.ndim == 3:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        
        return tensor * std + mean

class Normalize(Transform):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, tensor):
        return (tensor.astype(np.float32) - self.min_val) / (self.max_val - self.min_val)
    
    def invert(self, tensor, dtype=np.uint8):
        tensor = tensor.astype(np.float32) * (self.max_val - self.min_val) + self.min_val
        if not np.issubdtype(dtype, np.floating):
            info_int = np.iinfo(dtype)
            tensor = np.clip(tensor, info_int.min, info_int.max).astype(dtype)
        return tensor
    

class Flatten(Transform):
    def __call__(self, tensor):
        return tensor.flatten()
    
    
class OneHotEncode(Transform):
    def __init__(self, num_classes):
        self.enc = np.eye(num_classes, dtype=np.int32)
        
    def __call__(self, tensor):
        return self.enc[np.array(tensor)]
    
    
class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, tensor):
        if np.random.rand() < self.p:
            tensor = tensor[:, :, ::-1]

        return tensor
    
    
class RandomCrop(Transform):
    def __init__(self, size, padding=0):
        super().__init__()
        self.size = size
        self.padding = padding
        
    def __call__(self, tensor):
        if self.padding > 0:
            tensor = np.pad(tensor, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
         
        _, h, w = tensor.shape   
        i = np.random.randint(0, h - self.size + 1)
        j = np.random.randint(0, w - self.size + 1)
        
        return tensor[:, i : i + self.size, j : j + self.size]