import numpy as np

class LRScheduler:
    def __init__(self, opt):
        self.opt = opt
        self.base_lr = self.opt.lr
        self.last_epoch = 0
        
    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.opt.lr = lr
        
        return lr
    
    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, opt, step_size, gamma=0.1):
        super().__init__(opt)
        self.step_size = step_size
        self.gamma = gamma        
        
    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * factor
            

class MultiStepLR(LRScheduler):
    def __init__(self, opt, milestones, gamma=0.1):
        super().__init__(opt)
        self.milestones = set(milestones)
        self.gamma = gamma
        self._lr = self.base_lr
        
    def get_lr(self):
        if self.last_epoch in self.milestones:
            self._lr *= self.gamma
            
        return self._lr
            
            
class ExponentialLR(LRScheduler):
    def __init__(self, opt, gamma):
        super().__init__(opt)
        self.gamma = gamma
        
    def get_lr(self):
        return self.base_lr * (self.gamma ** self.last_epoch)
    
    
class CosineAnnealingLR(LRScheduler):
    def __init__(self, opt, T_max, eta_min=0.0):
        super().__init__(opt)
        self.T_max = T_max
        self.eta_min = eta_min
        
    def get_lr(self):
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max))
        
    
