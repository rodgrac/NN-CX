from nncx.tensor import Tensor
from nncx.utils import timeit

@timeit
def train(model, loss_fn, optimizer, dataloader, epochs):
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
                                
            for inputs, targets in dataloader[split]:
                with Tensor.no_grad(split == 'val'):        # Apply no_grad only when split is val
                    preds = model(inputs)
                    loss = loss_fn(preds, targets)
                
                epoch_loss[split] += float(loss.detach().data)
                
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                                        
                    optimizer.step()
                                            
            epoch_loss[split] /= len(dataloader[split])
        
        print(f"Epoch {epoch} => Train loss: {epoch_loss['train']:.4f}, Val loss: {epoch_loss['val']:.4f}")
            
            