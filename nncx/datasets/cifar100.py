import os
import pickle
import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_tar_extract

class CIFAR100Train(Dataset):
    def __init__(self):
        super().__init__()
        self.name = 'CIFAR100Train'
        
        self.download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        
        download_tar_extract(self.download_url, self.datasets_root)
        
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'train'), "rb") as fp:
            batch = pickle.load(fp, encoding='bytes')
        self.inputs = np.asarray(batch[b"data"])
        self.targets = np.asarray(batch[b"fine_labels"])


class CIFAR100Test(Dataset):
    def __init__(self):
        super().__init__()
        self.name = 'CIFAR100Test'
        
        self.download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        
        download_tar_extract(self.download_url, self.datasets_root)
        
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'test'), "rb") as fp:
            batch = pickle.load(fp, encoding='bytes')
        self.inputs = np.asarray(batch[b"data"])
        self.targets = np.asarray(batch[b"fine_labels"])