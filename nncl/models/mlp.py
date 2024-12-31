from nncl.models.model import Model
from nncl.layers.linear import Linear
from nncl.layers.activations import ReLU


class SineMLP(Model):
    def __init__(self) -> None:
        super().__init__()
        
        self.linear1 = Linear(1, 256)
        self.relu = ReLU()
        self.linear2 = Linear(256, 1)
                
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x