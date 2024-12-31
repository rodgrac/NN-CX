import numpy as np

from nncl.datasets.dataset import Dataset
from nncl.tensor import Tensor


class SineFn(Dataset):
    def __init__(self, num_samples) -> None:
        super().__init__()
        
        self.inputs = np.random.uniform(0, 2 * np.pi, size=num_samples)
        
        self.targets = np.sin(self.inputs)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return Tensor(self.inputs[idx], grad_en=True), Tensor(self.targets[idx])
