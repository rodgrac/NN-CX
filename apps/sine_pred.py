import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./')

from nncx.datasets.sinefn import Dataset, SineFn
from nncx.dataloader import DataLoader
from nncx.models.sinemlp import SineMLP
from nncx.losses import MSELoss
from nncx.optimizers import SGD
from nncx.trainer import train
from nncx.tensor import Tensor
from nncx.enums import BackendType


if __name__ == '__main__':
    backend_type = BackendType.CPU

    ds = SineFn(10000)
    train_idx, val_idx = Dataset.train_val_split_idxs(ds, val_split=0.2)
        
    dl = dict()
    dl['train'] = DataLoader(ds, backend_type=backend_type, batch_size=32, idxs=train_idx, shuffle=True)
    dl['val'] = DataLoader(ds, backend_type=backend_type, batch_size=32, idxs=val_idx, shuffle=False)
    
    m = SineMLP(backend_type=backend_type)
    # m.load_parameters('weights/sine_pred.npz')
            
    loss_fn = MSELoss()
    opt = SGD(m.parameters(), lr=0.0001)
    
    train(m, loss_fn, opt, dl, 10)
    
    m.save_parameters('weights/sine_pred.npz')
    
    x_test = np.linspace(0, 2* np.pi, 100)
    y_gt = np.sin(x_test)
        
    y_pred = m(Tensor(np.expand_dims(x_test, axis=-1), backend_type=backend_type))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_gt, label='Ground truth', color='blue')
    plt.plot(x_test, y_pred.get(), label='Model prediction', color='red')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()    