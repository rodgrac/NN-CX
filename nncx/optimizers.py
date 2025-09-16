class SGD:
    def __init__(self, backend, params, lr=0.01, momentum=0.9) -> None:
        self.backend = backend
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = {}
    
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            if param not in self.v:
                self.v[param] = self.backend.zeros(param.shape, param.dtype)
                
            v = self.v[param]
            
            v[:] = self.momentum * v - self.lr * param.grad
            param.data += v
            
            param._zero_grad()          
                      
    def zero_grad(self):
        for param in self.params:
            param._zero_grad()