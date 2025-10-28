from tqdm import tqdm

from nncx.tensor import Tensor
from nncx.utils import timeit

@timeit
def train(model, loss_fn, optimizer, dataloader, epochs, sched=None,
        accum_steps=4, patience=5
    ):
    assert type(model.backend) == type(dataloader['train'].backend), "Model and data need to be of same backend type!"
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        epoch_loss = {
            'train': 0,
            'val': 0
        }
        
        optimizer.zero_grad()
        for split in ['train', 'val']:
            if split=='train':
                model.train()
            else:
                model.eval()
                                
            for step, (inputs, targets) in enumerate(tqdm(dataloader[split])):
                with Tensor.no_grad(split == 'val'):        # Apply no_grad only when split is val
                    preds = model(inputs)
                    loss = loss_fn(preds, targets)
                
                epoch_loss[split] += float(loss.detach().data)
                
                if split == 'train':
                    loss = loss / accum_steps
                    loss.backward()
                    
                    if (step+1) % accum_steps == 0 or (step+1 == len(dataloader[split])):   
                        optimizer.step()
                        optimizer.zero_grad()
                                                                    
            epoch_loss[split] /= len(dataloader[split])

        print(f"[Trainer] Epoch {epoch}; LR={optimizer.lr:.4f} => Train loss: {epoch_loss['train']:.4f}, Val loss: {epoch_loss['val']:.4f}")
        
        # Early stopping
        if epoch_loss['val'] < best_val_loss:
            best_val_loss = epoch_loss['val']
            no_improve_epochs = 0
            model.save_parameters(f'weights/{model.__class__.__name__}/best_model.npz')
            print('[Trainer] Saved new best model')
        else:
            no_improve_epochs += 1
            print(f"[Trainer] No improvement in {no_improve_epochs} epochs")
            
        if no_improve_epochs >= patience:
            print(f"[Trainer] Early stopping at epoch {epoch}")
            break
        
        # Learning rate scheduler
        if sched is not None:
            sched.step()
            

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