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
        
    
class WarmupLR(LRScheduler):
    def __init__(self, opt, base_scheduler, warmup_epochs, warmup_start_lr=0.0):
        super().__init__(opt)
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self._finished_warmup = False
        
    def step(self):
        #Warmup
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
            alpha = self.last_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
            self.opt.lr = lr
            return lr
        
        # After warmup
        if not self._finished_warmup:
            if isinstance(self.base_scheduler, CosineAnnealingLR):
                self.base_scheduler.T_max -= self.warmup_epochs
            self._finished_warmup = True
        
        return self.base_scheduler.step()