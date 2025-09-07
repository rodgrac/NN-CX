import os
import pickle
import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_tar_extract

class CIFAR100Train(Dataset):
    def __init__(self, label_type='fine'):
        super().__init__()
        self.name = 'CIFAR100Train'
        self.image_size = (3, 32, 32)
        
        self.download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        
        download_tar_extract(self.download_url, self.datasets_root)
    
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'train'), "rb") as fp:
            batch = pickle.load(fp, encoding='bytes')
        self.inputs = np.asarray(batch[b"data"], dtype=np.uint8).reshape((-1,) + self.image_size)
        
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'meta'), "rb") as fp:
            metadata = pickle.load(fp, encoding='latin1')
        
        if label_type == 'fine':
            self.targets = np.asarray(batch[b"fine_labels"])
            self.label_names = metadata['fine_label_names']
        else:
            self.targets = np.asarray(batch[b"coarse_labels"])
            self.label_names = metadata['coarse_label_names']  
            
        self.num_labels = len(self.label_names)  
        self.data_mean = [0.5071, 0.4865, 0.4409]
        self.data_std = [0.2673, 0.2564, 0.2762]      


class CIFAR100Test(Dataset):
    def __init__(self, label_type='fine'):
        super().__init__()
        self.name = 'CIFAR100Test'
        self.image_size = (32, 32, 3)
        
        self.download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        
        download_tar_extract(self.download_url, self.datasets_root)
        
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'test'), "rb") as fp:
            batch = pickle.load(fp, encoding='bytes')
        self.inputs = np.asarray(batch[b"data"], dtype=np.uint8).reshape((-1,) + self.image_size).transpose(0, 3, 1, 2) # CIFAR test split is in NHWC
        
        with open(os.path.join(self.datasets_root, self.download_url.split('/')[-1].split('.')[0], 'meta'), "rb") as fp:
            metadata = pickle.load(fp, encoding='latin1')
        
        if label_type == 'fine':
            self.targets = np.asarray(batch[b"fine_labels"])
            self.label_names = metadata['fine_label_names']
        else:
            self.targets = np.asarray(batch[b"coarse_labels"])
            self.label_names = metadata['coarse_label_names']
            
        self.num_labels = len(self.label_names)   
        self.data_mean = [0.5071, 0.4865, 0.4409]
        self.data_std = [0.2673, 0.2564, 0.2762]      