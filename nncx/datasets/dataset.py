import os
from abc import ABC, abstractmethod
import numpy as np

from nncx.config import PROJECT_ROOT
from nncx.tensor import Tensor

class Dataset(ABC):
    def __init__(self):
        super().__init__()
                
        self.datasets_root = os.path.join(PROJECT_ROOT, 'data')
        os.makedirs(self.datasets_root, exist_ok=True)
        
        self.inputs = None
        self.targets = None
        
        self.transform = []
    
    
    def __len__(self):
        return len(self.inputs)
    
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        for transform in self.transforms:
            data_item = transform(data_item)
        
        return Tensor(data_item, backend=backend, grad_en=True), \
                Tensor(target_item, backend=backend)
        
        
    def set_transforms(self, transforms):
        self.transforms = transforms
        
    def get_stats(self, stats='minmax'):
        return get_stats(self.inputs, stats)
    
    def split(self, train_ratio=0.8, shuffle=True, test_set=True, seed=None):
        idxs = np.arange(len(self.inputs))
        
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(idxs)
            
        split_idx = int(len(idxs) * train_ratio)
        train_idxs = idxs[:split_idx]
        train_subset = SubDataset(self, train_idxs)
        
        if test_set:
            val_test_split_idx = (len(idxs) - split_idx) // 2
            val_idxs = idxs[split_idx: split_idx + val_test_split_idx]
            test_idxs = idxs[split_idx + val_test_split_idx:]

            val_subset = SubDataset(self, val_idxs)
            test_subset = SubDataset(self, test_idxs)
            
            print(f"Total samples: {len(idxs)}, Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
            
            return train_subset, val_subset, test_subset
        else:
            val_idxs = idxs[split_idx:]
            val_subset = SubDataset(self, val_idxs)
            
            print(f"Total samples: {len(idxs)}, Train: {len(train_subset)}, Val: {len(val_subset)}")
            
            return train_subset, val_subset
    

class SubDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
        self.transforms = []
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item, target_item = self.dataset[(self.indices[idx], backend)]
        
        with Tensor.no_grad():
            for transform in self.transforms:
                data_item = transform(data_item)
            
        return data_item, target_item
        
    
    def set_transforms(self, transforms):
        self.transforms = transforms
        
    def get_stats(self, stats='minmax'):
        return get_stats(self.dataset.inputs, stats, self.indices)
        
        
def get_stats(data, stats='min-max', idxs=None):
    if idxs is None:
        idxs = np.arange(len(data))
    
    if stats == 'min-max':
        return np.min(data[idxs]), np.max(data[idxs])
    elif stats == 'mean-std':
        return np.mean(data[idxs]), np.std(data[idxs])
    else:
        raise NotImplementedError