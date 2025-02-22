import os
import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_tar_extract


class CaliforniaHousing(Dataset):
    def __init__(self):
        super().__init__()
        self.name = 'CaliforniaHousing'
        self.download_url = 'https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz'
        self.data_file = 'cal_housing.data'
                        
        if not os.path.exists(os.path.join(self.datasets_root, self.name, self.data_file)):
            download_tar_extract(self.download_url, self.datasets_root)
        
        data = np.loadtxt(os.path.join(self.datasets_root, self.name, self.data_file), delimiter=',')
        
        self.inputs = data[:, :-1]
        self.targets = np.expand_dims(data[:, -1], axis=-1)
        
        