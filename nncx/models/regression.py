from nncx.models.model import Model
from nncx.layers.linear import Linear
from nncx.layers.activations import ReLU


class Regression(Model):
    def __init__(self, input_dim, backend):
        super().__init__(backend)
        
        self.linear1 = Linear(input_dim, 64, backend=backend)
        self.linear2 = Linear(64, 32, backend=backend)
        self.linear3 = Linear(32, 1, backend=backend)
        self.relu = ReLU()
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        
        return x