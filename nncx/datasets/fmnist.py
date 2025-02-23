import os
import gzip
import numpy as np

from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_tar_extract


class FashionMNISTTrain(Dataset):
    def __init__(self):
        super().__init__()
        self.name = 'FashionMNISTTrain'
        self.num_labels = 10
        self.label_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
        
        self.download_url = {
            'inputs': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        }
        
        for url in self.download_url.values():
            download_tar_extract(url, self.datasets_root, extract=False)
                
        with gzip.open(os.path.join(self.datasets_root, self.download_url['inputs'].split('/')[-1]), 'rb') as f:
            f.read(16)
            self.inputs = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            
        with gzip.open(os.path.join(self.datasets_root, self.download_url['labels'].split('/')[-1]), 'rb') as f:
            f.read(8)
            self.targets = np.frombuffer(f.read(), dtype=np.uint8)

                

class FashionMNISTTest(Dataset):  
    def __init__(self):
        super().__init__()
        self.name = 'FashionMNISTTest'
        self.num_labels = 10
        self.label_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

        self.download_url = {
            'inputs': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
        }
        
        for url in self.download_url.values():
            download_tar_extract(url, self.datasets_root, extract=False)
                
        with gzip.open(os.path.join(self.datasets_root, self.download_url['inputs'].split('/')[-1]), 'rb') as f:
            f.read(16)
            self.inputs = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            
        with gzip.open(os.path.join(self.datasets_root, self.download_url['labels'].split('/')[-1]), 'rb') as f:
            f.read(8)
            self.targets = np.frombuffer(f.read(), dtype=np.uint8)
        