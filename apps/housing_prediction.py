from nncx.backend.utils import init_backend
from nncx.datasets.california_housing import CaliforniaHousing
from nncx.datasets.transform import Standardize
from nncx.dataloader import DataLoader
from nncx.enums import BackendType
from nncx.models.regression import Regression
from nncx.losses import MSELoss
from nncx.optimizers import SGD
from nncx.trainer import train, evaluate
from nncx.visualizer import visualize_predictions

if __name__ == '__main__':
    do_train = True
    
    backend = init_backend(BackendType.CPU)
    
    ds = CaliforniaHousing()
    train_ds, val_ds, test_ds = ds.split(train_ratio=0.8, shuffle=True, test_set=True, seed=42)
    
    train_mean_x, train_std_x = train_ds.get_input_stats('mean-std', axis=0)
    train_mean_y, train_std_y = train_ds.get_target_stats('mean-std', axis=0)
    
    train_transforms_x = [Standardize(train_mean_x, train_std_x)]
    train_transforms_y = [Standardize(train_mean_y, train_std_y)]
    
    ds.set_transforms(train_transforms_x, train_transforms_y)
        
    dl = dict()
    dl['train'] = DataLoader(train_ds, backend=backend, batch_size=32, shuffle=True)
    dl['val'] = DataLoader(val_ds, backend=backend, batch_size=32, shuffle=False)
    dl['test'] = DataLoader(test_ds, backend=backend, batch_size=32, shuffle=False)
    
    first_batch = next(iter(dl['train']))
    
    model = Regression(first_batch[0].shape[-1], backend=backend)
    
    loss_fn = MSELoss()

    if do_train:
        opt = SGD(model.parameters(), lr=0.001)
        train(model, loss_fn, opt, dl, 25)
        
        model.save_parameters('weights/housing_prediction.npz')
        
    else:
        model.load_parameters('weights/housing_prediction.npz')
    
    preds, targets = evaluate(model, loss_fn, dl)
    
    visualize_predictions(preds, targets, title='Predicted vs Actual House Pricing', xlabel='Actual Prices', ylabel='Predicted Prices')
    
    
    
    
    

