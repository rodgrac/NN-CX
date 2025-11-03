from tqdm import tqdm

from nncx.tensor import Tensor
from nncx.utils import timeit

@timeit
def train(model, loss_fn, optimizer, dataloader, epochs, sched=None,
        accum_steps=4, patience=5
    ):
    assert type(model.backend_type) == type(dataloader['train'].backend_type), "Model and data need to be of same backend type!"
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        epoch_loss = {}
        
        optimizer.zero_grad()
        for split in ['train', 'val']:
            if split=='train':
                model.train()
            else:
                model.eval()
            
            global_steps = 0
            for step, (inputs, targets) in enumerate(tqdm(dataloader[split])):
                with Tensor.no_grad(split == 'val'):        # Apply no_grad only when split is val
                    preds = model(inputs)
                    loss = loss_fn(preds, targets)
                    total_loss = loss[0]
                
                if split not in epoch_loss:
                    epoch_loss[split] = (0.0, ) * len(loss)
                    
                epoch_loss[split] = tuple(a + b for a, b in zip(epoch_loss[split], (float(l.detach().data) for l in loss)))                    
                
                if split == 'train':
                    total_loss = total_loss / accum_steps
                    total_loss.backward()
                    
                    if (step+1) % accum_steps == 0 or (step+1 == len(dataloader[split])):   
                        optimizer.step()
                        optimizer.zero_grad()
                
                # print(f"load+queue {t1-t0:.3f}s | fwd+loss {t2-t1:.3f}s | bwd+step {t3-t2:.3f}s")
                global_steps += 1
                                                                    
            epoch_loss[split] = tuple(l / len(dataloader[split]) for l in epoch_loss[split])

        print(
            f"[Trainer] Epoch {epoch}; LR={optimizer.lr:.6f} => "
            f"Train loss: {tuple(f'{l:.4f}' for l in epoch_loss['train'])}, "
            f"Val loss: {tuple(f'{l:.4f}' for l in epoch_loss['val'])}"
        )
        
        # Early stopping
        if epoch_loss['val'][0] < best_val_loss:
            best_val_loss = epoch_loss['val'][0]
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
    assert type(model.backend_type) == type(dataloader['test'].backend_type), "Model and data need to be of same backend type!"
    
    model.eval()
    
    preds_all = []
    targets_all = [] 
    test_loss = None
    for inputs, targets in tqdm(dataloader['test']):
        with Tensor.no_grad():        
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            
        if test_loss is None:
            test_loss = (0.0, ) * len(loss)
            
        test_loss = tuple(a + b for a, b in zip(test_loss, (float(l.detach().data) for l in loss)))   
                
        preds_all.extend(preds)
        targets_all.extend(targets)
    
    test_loss = tuple(l / len(dataloader['test']) for l in test_loss)                                    
    
    print(f"[Trainer] Test loss: {tuple(f'{l:.4f}' for l in test_loss)}")
    
    return preds_all, targets_all