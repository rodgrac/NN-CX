from nncx.models.model import Model
from nncx.layers import *

class FaceDetector(Model):
    def __init__(self, backend, in_size=224):
        super().__init__(backend)
        self.in_size = in_size
        
        self.features = Sequential(
            Conv2d(3, 32, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            Conv2d(32, 32, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            MaxPool2d(backend, kernel_size=2, stride=2),
            
            Conv2d(32, 64, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            Conv2d(64, 64, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            MaxPool2d(backend, kernel_size=2, stride=2),
            
            Conv2d(64, 128, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            Conv2d(128, 128, backend, kernel_size=3, stride=1, pad=True),
            ReLU(),
            MaxPool2d(backend, kernel_size=2, stride=2),            
        )
        
        dummy = Tensor(shape=(1, 3, in_size, in_size), backend=backend)
        num_features = self.features(dummy).reshape((1, -1)).shape[1]
        
        self.cls_head = Sequential(
            Linear(num_features, 128, backend),
            ReLU(),
            Linear(128, 1, backend)
        )
        
        self.box_head = Sequential(
            Linear(num_features, 128, backend),
            ReLU(),
            Linear(128, 4, backend),
            Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x).reshape((x.shape[0], -1))
        return self.box_head(x), self.cls_head(x)