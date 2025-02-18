class SGD:
    def __init__(self, params, lr=0.01) -> None:
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
                param._zero_grad()          
                      
    def zero_grad(self):
        for param in self.params:
            param._zero_grad()