from nncl.models.model import Model
from nncl.layers.linear import Linear
from nncl.layers.activations import Tanh


class SineMLP(Model):
    def __init__(self) -> None:
        super().__init__()
        
        self.linear1 = Linear(1, 32)
        self.linear2 = Linear(32, 32)
        self.linear3 = Linear(32, 1)
        self.tanh = Tanh()
        
                
    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        
        return x