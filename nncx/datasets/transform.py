import numpy as np
from PIL import Image

from nncx.datasets.dataset import Dataset

class Transform:
    def __call__(self, input_tensor, target_tensor, target_type):
        raise NotImplementedError
    
class Standardize(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        
    def __call__(self, input_tensor, target_tensor, target_type):
        mean = self.mean
        std = self.std
        if input_tensor.ndim == 3:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        
        return (input_tensor - mean) / std, target_tensor
    
    def invert(self, input_tensor):
        mean = self.mean
        std = self.std
        if input_tensor.ndim == 3:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        
        return input_tensor * std + mean

class Normalize(Transform):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, input_tensor, target_tensor, target_type):
        return (input_tensor.astype(np.float32) - self.min_val) / (self.max_val - self.min_val), target_tensor
    
    def invert(self, input_tensor, dtype=np.uint8):
        input_tensor = input_tensor.astype(np.float32) * (self.max_val - self.min_val) + self.min_val
        if not np.issubdtype(dtype, np.floating):
            info_int = np.iinfo(dtype)
            input_tensor = np.clip(input_tensor, info_int.min, info_int.max).astype(dtype)
        return input_tensor
    

class Flatten(Transform):
    def __call__(self, input_tensor, target_tensor, target_type):
        return input_tensor.flatten(), target_tensor
    
    
class OneHotEncode(Transform):
    def __init__(self, num_classes):
        self.enc = np.eye(num_classes, dtype=np.int32)
        
    def __call__(self, target_tensor):
        return self.enc[np.array(target_tensor)]
    
    
class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, input_tensor, target_tensor, target_type):
        if np.random.rand() < self.p:
            input_tensor = input_tensor[:, :, ::-1]
            if target_type==Dataset.TargetType.BBOX:
                cx, cy, w, h = target_tensor
                cx = 1.0 - cx
                target_tensor = np.array([cx, cy, w, h], dtype=np.float32)

        return input_tensor, target_tensor
    
class RandomCrop(Transform):
    def __init__(self, size, padding=0):
        super().__init__()
        self.size = size
        self.padding = padding
        
    def __call__(self, input_tensor, target_tensor, target_type):
        if self.padding > 0:
            input_tensor = np.pad(input_tensor, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
         
        _, h, w = input_tensor.shape   
        i = np.random.randint(0, h - self.size + 1)
        j = np.random.randint(0, w - self.size + 1)
        
        return input_tensor[:, i : i + self.size, j : j + self.size], target_tensor
    
