import math
import numpy as np

from nncx.tensor import Tensor

class DataLoader:
    def __init__(self, dataset, backend, batch_size, shuffle=True) -> None:
        self.dataset = dataset
        self.backend = backend
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.idxs = np.arange(len(self.dataset))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
            
        for i in range(0, len(self.idxs), self.batch_size):
            batch = [self.dataset[(idx, self.backend)] for idx in self.idxs[i:i+self.batch_size]]
            
            yield self._collate(batch)
            
    def __len__(self):
        if self.idxs is None:
            return math.ceil(len(self.dataset)/float(self.batch_size))
        else:
            return math.ceil(len(self.idxs) / float(self.batch_size))
        
    def _collate(self, batch):    
        batch_tensors = []    
        for items in zip(*batch):
            if isinstance(items[0], (tuple, list)):
                items = tuple(Tensor.stack([t[i] for t in items]) for i in range(len(items[0])))
            else:
                items = Tensor.stack(items)
                
            batch_tensors.append(items)
        
        return tuple(batch_tensors)