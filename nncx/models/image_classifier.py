from nncx.models.model import Model
from nncx.layers import *

class ImageClassifier(Model):
    def __init__(self, backend_type, out_features, num_classes):
        super().__init__(backend_type)
        
        self.features = Sequential(
            Conv2d(3, 64, backend_type, kernel_size=3, stride=1, pad=True),
            ReLU(),
            MaxPool2d(backend_type, kernel_size=2, stride=2),
            Conv2d(64, 128, backend_type, kernel_size=3, stride=1, pad=True),
            ReLU(),
            MaxPool2d(backend_type, kernel_size=2, stride=2),
            Conv2d(128, 256, backend_type, kernel_size=3, stride=1, pad=True),
            ReLU(),
        )
            
        self.fc = Sequential(
            Linear(out_features, 512, backend_type),
            ReLU(),
            Linear(512, num_classes, backend_type)
        )
        
    def forward(self, x):
        b, *_ = x.shape
        x = self.features(x).reshape((b, -1))
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        with Tensor.no_grad():
            x = self.forward(x)
            return x.argmax(axis=-1)