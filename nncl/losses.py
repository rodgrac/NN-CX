import numpy as np

from nncl.tensor import Tensor


class MSELoss:
    def __call__(self, pred, target):
        loss = Tensor(np.mean((pred.data - target.data)**2), grad_en=pred.grad_en)
        
        if loss.grad_en:
            loss.grad_fn = lambda grad: [grad * 2 * (pred.data - target.data) / target.size]
            loss._prev = [pred] 
                
        return loss