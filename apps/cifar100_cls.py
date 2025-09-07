from nncx.backend.utils import init_backend
from nncx.datasets.cifar100 import CIFAR100Train, CIFAR100Test
from nncx.dataloader import DataLoader
from nncx.datasets.transform import Normalize, Standardize, OneHotEncode
from nncx.models.image_classifier import ImageClassifier
from nncx.losses import CrossEntropyLoss
from nncx.optimizers import SGD
from nncx.trainer import train, evaluate
from nncx.metrics import ClassificationMetrics
from nncx.enums import BackendType
import nncx.visualizer as viz


if __name__ == '__main__':
    do_train = False
    batch_size = 1024
    epochs = 10
    
    backend = init_backend(BackendType.GPU)
    
    train_val_ds = CIFAR100Train(label_type='fine')
    test_ds = CIFAR100Test(label_type='fine')
    train_ds, val_ds = train_val_ds.split(test_set=False, seed=42)
        
    viz.view_image_dataset(train_ds)
    
    transforms_x = [Normalize(min_val=0, max_val=255.0), 
                    Standardize(test_ds.data_mean, test_ds.data_std)]
    transforms_y = [OneHotEncode(test_ds.num_labels)]
    
    train_val_ds.set_transforms(transforms_x, transforms_y)
    test_ds.set_transforms(transforms_x, transforms_y)
    
    dl = dict()
    dl['train'] = DataLoader(train_ds, backend=backend, batch_size=batch_size, shuffle=True)
    dl['val'] = DataLoader(val_ds, backend=backend, batch_size=batch_size, shuffle=False)
    dl['test'] = DataLoader(test_ds, backend=backend, batch_size=batch_size, shuffle=False)

    model = ImageClassifier(backend, out_features=32*8*8, num_classes=test_ds.num_labels)
    
    loss_fn = CrossEntropyLoss()
    
    if do_train:
        opt = SGD(model.parameters(), lr=0.05)
        train(model, loss_fn, opt, dl, epochs)
        
        model.save_parameters('weights/cifar100_cls.npz')
    else:
        model.load_parameters('weights/cifar100_cls.npz')
        
    preds, targets = evaluate(model, loss_fn, dl)
    
    cls_metrics = ClassificationMetrics()
    
    cls_metrics.accuracy(preds, targets)
    cls_metrics.precision_recall_f1(preds, targets, num_classes=test_ds.num_labels)
    
    viz.plot_confusion_matrix(preds, targets, num_classes=test_ds.num_labels, class_names=test_ds.label_names)
    
    viz.visualize_predictions(model, dl['test'])
    