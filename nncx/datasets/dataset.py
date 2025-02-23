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
        
        self.num_labels = None
        self.label_names = None
        
        self.transforms_inputs = []
        self.transforms_targets = []
    
    
    def __len__(self):
        return len(self.inputs)
    
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        for transform in self.transforms_inputs:
            data_item = transform(data_item)
            
        for transform in self.transforms_targets:
            target_item = transform(target_item)
        
        return Tensor(data_item, backend=backend, grad_en=True), \
                Tensor(target_item, backend=backend)
        
        
    def set_transforms(self, transforms_inputs, transforms_targets=[]):
        self.transforms_inputs = transforms_inputs
        self.transforms_targets = transforms_targets
        
    def get_input_stats(self, stats='min-max', axis=-1):
        return get_stats(self.inputs, stats, axis=axis)
    
    def get_target_stats(self, stats='min-max', axis=-1):
        return get_stats(self.targets, stats, axis=axis)
    
    
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
            
            print(f"[SPLIT] Total samples: {len(idxs)}, Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
            
            return train_subset, val_subset, test_subset
        else:
            val_idxs = idxs[split_idx:]
            val_subset = SubDataset(self, val_idxs)
            
            print(f"[SPLIT] Total samples: {len(idxs)}, Train: {len(train_subset)}, Val: {len(val_subset)}")
            
            return train_subset, val_subset
    

class SubDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
        self.transforms_inputs = []
        self.transforms_targets = []
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item, target_item = self.dataset[(self.indices[idx], backend)]
        
        with Tensor.no_grad():
            for transform in self.transforms_inputs:
                data_item = transform(data_item)
            
            for transform in self.transforms_targets:
                target_item = transform(target_item)
            
        return data_item, target_item
        
    
    def set_transforms(self, transforms_inputs, transforms_targets=[]):
        self.transforms_inputs = transforms_inputs
        self.transforms_targets = transforms_targets
        
    def get_input_stats(self, stats='min-max', axis=None):
        return get_stats(self.dataset.inputs, stats, self.indices, axis=axis)
    
    def get_target_stats(self, stats='min-max', axis=None):
        return get_stats(self.dataset.targets, stats, self.indices, axis=axis)
        
        
def get_stats(data, stats='min-max', idxs=None, axis=-1):
    if idxs is None:
        idxs = np.arange(len(data))
    
    if stats == 'min-max':
        return np.min(data[idxs], axis=axis), np.max(data[idxs], axis=axis)
    elif stats == 'mean-std':
        return np.mean(data[idxs], axis=axis), np.std(data[idxs], axis=axis) + 1e-8
    else:
        raise NotImplementedError