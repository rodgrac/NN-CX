from tqdm import tqdm

from nncx.tensor import Tensor
from nncx.utils import timeit

@timeit
def train(model, loss_fn, optimizer, dataloader, epochs, sched=None):
    assert type(model.backend) == type(dataloader['train'].backend), "Model and data need to be of same backend type!"
    
    for epoch in range(epochs):
        epoch_loss = {
            'train': 0,
            'val': 0
        }
        for split in ['train', 'val']:
            if split=='train':
                model.train()
            else:
                model.eval()
                                
            for inputs, targets in tqdm(dataloader[split]):
                with Tensor.no_grad(split == 'val'):        # Apply no_grad only when split is val
                    preds = model(inputs)
                    loss = loss_fn(preds, targets)
                
                epoch_loss[split] += float(loss.detach().data)
                
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                                        
                    optimizer.step()
                    if sched is not None:
                        sched.step()
                                            
            epoch_loss[split] /= len(dataloader[split])
        
        print(f"[Trainer] Epoch {epoch}; LR={optimizer.lr:.4f} => Train loss: {epoch_loss['train']:.4f}, Val loss: {epoch_loss['val']:.4f}")
            

@timeit
def evaluate(model, loss_fn, dataloader):
    assert type(model.backend) == type(dataloader['test'].backend), "Model and data need to be of same backend type!"
    
    model.eval()
    
    preds_all = []
    targets_all = [] 
    test_loss = 0    
    for inputs, targets in tqdm(dataloader['test']):
        with Tensor.no_grad():        
            preds = model(inputs)
            loss = loss_fn(preds, targets)
        
        test_loss += float(loss.detach().data)
        
        preds_all.extend(preds.get())
        targets_all.extend(targets.get())
                                    
    test_loss /= len(dataloader['test'])
    
    print(f"[Trainer] Test loss: {test_loss:.4f}")
    
    return preds_all, targets_all