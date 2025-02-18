import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.tensor import Tensor


class SineFn(Dataset):
    def __init__(self, num_samples) -> None:
        super().__init__()
        
        self.inputs = np.random.uniform(0, 2 * np.pi, size=num_samples)
        
        self.targets = np.sin(self.inputs)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, key):
        idx, backend = key
        return Tensor(self.inputs[idx], backend=backend, grad_en=True), Tensor(self.targets[idx], backend=backend)
