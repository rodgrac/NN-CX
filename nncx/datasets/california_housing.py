import os
import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.tensor import Tensor
from nncx.datasets.utils import download_tar_extract


class CaliforniaHousing(Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self.name = 'CaliforniaHousing'
        self.download_url = 'https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz'
        self.data_file = 'cal_housing.data'
        
        self.transforms = transforms or []
        
        if not os.path.exists(os.path.join(self.datasets_root, self.name, self.data_file)):
            download_tar_extract(self.download_url, self.datasets_root)
        
        data = np.loadtxt(os.path.join(self.datasets_root, self.name, self.data_file), delimiter=',')
        
        self.inputs = data[:, :-1]
        self.targets = data[:, -1]
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        for transform in self.transforms:
            data_item = transform(data_item)
        
        return Tensor(data_item, backend=backend, grad_en=True), \
                Tensor(target_item, backend=backend)
        