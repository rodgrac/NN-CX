from nncx.tensor import Tensor
from nncx.enums import DataType
from nncx.layers.activations import SoftMax, Sigmoid


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
    
    
class BCEWithLogitsLoss:
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def __call__(self, pred, target):
        backend = pred.backend
                
        # mean(max(x,0) − x⋅y + log(1+e−∣x∣))
        max_val = backend.maximum(pred.data, 0) 
        loss_val = backend.mean(max_val - pred.data * target.data + backend.log(1 + backend.exp(-backend.abs(pred.data))))
        
        loss = Tensor(loss_val, backend=backend, grad_en=pred.grad_en)
        
        if loss.grad_en:
            with Tensor.no_grad():
                sigmoid_out = self.sigmoid(pred).data
                sigmoid_out = backend.clip(sigmoid_out, 1e-7, 1 - 1e-7)
                
            loss.grad_fn = lambda grad: [grad * (sigmoid_out - target.data) / pred.shape[0]]
            loss._prev = [pred]
            
        return loss
    

class SmoothL1Loss:
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def __call__(self, pred, target, mask=None):
        backend = pred.backend
        
        diff = pred.data - target.data
        abs_diff = backend.abs(diff)
        
        smooth_mask = backend.astype(abs_diff < self.beta, DataType.FLOAT32)
        loss_val = smooth_mask * (0.5 * diff**2 / self.beta) + (1 - smooth_mask) * (abs_diff - 0.5 * self.beta)
        
        if mask is not None:
            loss_val *= mask.data
            mask_norm = backend.sum(mask.data) + 1e-6
            loss_val = backend.sum(loss_val) / mask_norm
        else:
            loss_val = backend.mean(loss_val)
        
        loss = Tensor(loss_val, backend=backend, grad_en=pred.grad_en)
                
        if loss.grad_en:
            grad_val = smooth_mask * (diff / self.beta) + (1 - smooth_mask) * backend.sign(diff)
            if mask is not None:
                grad_val *= mask.data
                grad_val /= mask_norm
            else:
                grad_val /= pred.shape[0]
            
            loss.grad_fn = lambda grad: [grad * grad_val]
            loss._prev = [pred]
            
        return loss
        