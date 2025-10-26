from nncx.backend.utils import init_backend
from nncx.datasets.wider_face import WIDERFace
from nncx.dataloader import DataLoader
from nncx.models.regression import Regression
from nncx.losses import MSELoss
from nncx.optimizers import SGD
from nncx.trainer import train, evaluate
from nncx.enums import BackendType
import nncx.visualizer as viz


if __name__ == '__main__':
    backend = init_backend(BackendType.GPU)
    
    train_val_ds = WIDERFace(split='train')
    test_ds = WIDERFace(split='val')
    
    train_ds, val_ds = train_val_ds.split_dataset(test_set=False, seed=42)
    
    viz.view_image_dataset(train_ds, backend)
    
    
    
    