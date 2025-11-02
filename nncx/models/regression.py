from nncx.models.model import Model
from nncx.layers.linear import Linear
from nncx.layers.activations import ReLU


class Regression(Model):
    def __init__(self, input_dim, backend_type):
        super().__init__(backend_type)
        
        self.linear1 = Linear(input_dim, 64, backend_type=backend_type)
        self.linear2 = Linear(64, 32, backend_type=backend_type)
        self.linear3 = Linear(32, 1, backend_type=backend_type)
        self.relu = ReLU()
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        
        return x