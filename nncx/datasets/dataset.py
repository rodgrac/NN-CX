import os
from abc import ABC, abstractmethod
import numpy as np

from nncx.config import PROJECT_ROOT

class Dataset(ABC):
    def __init__(self):
        super().__init__()
        
        self.datasets_root = os.path.join(PROJECT_ROOT, 'data')
        os.makedirs(self.datasets_root, exist_ok=True)
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, key):
        # Needs to return a Tensor tuple of form: {inputs, targets}
        pass
    
    
    @staticmethod
    def train_val_split_idxs(dataset, val_split=0.2, shuffle=True):
        idxs = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(idxs)
        
        split = int((1 - val_split) * len(dataset))
        
        print(f"Total samples: {len(dataset)}, Train: {split}, Val: {len(dataset) - split}")
        return idxs[:split], idxs[split:]