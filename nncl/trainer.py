from nncl.tensor import Tensor

def train(model, loss_fn, optimizer, dataloader, epochs):
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
        
        print(f"Epoch {epoch} => Train loss: {epoch_loss['train']}, Val loss: {epoch_loss['val']}")
            
            