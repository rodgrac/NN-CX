class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        
        self._params = []
        for layer in self.layers:
            if hasattr(layer, '_params'):
                self._params.extend(layer._params)
                
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    