import math
import numpy as np

from nncx.tensor import Tensor

class DataLoader:
    def __init__(self, dataset, backend, batch_size, idxs=None, shuffle=True) -> None:
        self.dataset = dataset
        self.backend = backend
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxs = idxs
        
    def __iter__(self):
        if self.idxs is None:
            self.idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.idxs)
            
        for i in range(0, len(self.idxs), self.batch_size):
            batch_inputs = []
            batch_targets = []
            for idx in self.idxs[i:i+self.batch_size]:
                input_t, target_t = self.dataset[(idx, self.backend)]
                batch_inputs.append(input_t)
                batch_targets.append(target_t)
                            
            yield Tensor.stack(batch_inputs), Tensor.stack(batch_targets)
            
    def __len__(self):
        if self.idxs is None:
            return math.ceil(len(self.dataset)/float(self.batch_size))
        else:
            return math.ceil(len(self.idxs) / float(self.batch_size))
        
        