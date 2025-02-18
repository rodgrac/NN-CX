from nncx.models.model import Model
from nncx.layers.linear import Linear
from nncx.layers.activations import Tanh


class SineMLP(Model):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        
        self.linear1 = Linear(1, 32, backend=backend)
        self.linear2 = Linear(32, 32, backend=backend)
        self.linear3 = Linear(32, 1, backend=backend)
        self.tanh = Tanh()
        
                
    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        
        return x