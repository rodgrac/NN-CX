import numpy as np

from nncx.tensor import Tensor
from nncx.layers.activations import SoftMax


class MSELoss:
    def __call__(self, pred, target):
        backend = pred.backend
        loss = Tensor(backend.sum((pred.data - target.data)**2, axis=None, keepdims=False) / pred.size,
                      backend=backend,
                      grad_en=pred.grad_en)
        
        if loss.grad_en:
            loss.grad_fn = lambda grad: [grad * 2 * (pred.data - target.data) / target.size]
            loss._prev = [pred] 
                
        return loss
    
    
class CrossEntropyLoss:
    def __init__(self):
        self.softmax = SoftMax()
    
    def __call__(self, pred, target):
        backend = pred.backend
        
        with Tensor.no_grad():
            softmax_out = self.softmax(pred).data
        
        loss = Tensor(-backend.sum(target.data * backend.log(softmax_out + 1e-9)) / pred.shape[0],
                      backend=backend,
                      grad_en=pred.grad_en)
        
        if loss.grad_en:
            loss.grad_fn = lambda grad: [grad * (softmax_out - target.data) / pred.shape[0]]
            loss._prev = [pred]
            
        return loss