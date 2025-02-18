from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):
    
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