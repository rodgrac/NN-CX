from nncx.tensor import Tensor
from nncx.models.model import Model
from nncx.layers.linear import Linear
from nncx.layers.activations import ReLU


class Classifier(Model):
    def __init__(self, input_dim, num_classes, backend_type, hidden_dim=128):
        super().__init__(backend_type)
        
        self.linear1 = Linear(input_dim, hidden_dim, backend_type=backend_type)
        self.linear2 = Linear(hidden_dim, num_classes, backend_type=backend_type)
        self.relu = ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x
    
    
    def predict(self, x):
        with Tensor.no_grad():
            x = self.forward(x)
            return x.argmax(axis=-1)
        