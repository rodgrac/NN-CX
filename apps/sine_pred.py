import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./')

from nncl.backend.utils import init_backend
from nncl.datasets.sinefn import Dataset, SineFn
from nncl.dataloader import DataLoader
from nncl.models.sinemlp import SineMLP
from nncl.losses import MSELoss
from nncl.optimizers import SGD
from nncl.trainer import train
from nncl.tensor import Tensor
from nncl.enums import BackendType


if __name__ == '__main__':
    backend = init_backend(BackendType.CPU)

    ds = SineFn(10000)
    train_idx, val_idx = Dataset.train_val_split_idxs(ds, val_split=0.2)
        
    dl = dict()
    dl['train'] = DataLoader(ds, backend=backend, batch_size=32, idxs=train_idx, shuffle=True)
    dl['val'] = DataLoader(ds, backend=backend, batch_size=32, idxs=val_idx, shuffle=False)
    
    m = SineMLP(backend=backend)
    # m.load_parameters('weights/sine_pred.npz')
            
    loss_fn = MSELoss()
    opt = SGD(m.parameters(), lr=0.0001)
    
    train(m, loss_fn, opt, dl, 10)
    
    m.save_parameters('weights/sine_pred.npz')
    
    x_test = np.linspace(0, 2* np.pi, 100)
    y_gt = np.sin(x_test)
        
    y_pred = m(Tensor(np.expand_dims(x_test, axis=-1), backend=backend))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_gt, label='Ground truth', color='blue')
    plt.plot(x_test, y_pred.get(), label='Model prediction', color='red')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()    