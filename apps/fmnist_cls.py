from nncx.datasets.fmnist import FashionMNISTTrain, FashionMNISTTest
from nncx.datasets.transform import Normalize, Flatten, OneHotEncode
from nncx.dataloader import DataLoader
from nncx.models.classifier import Classifier
from nncx.losses import CrossEntropyLoss
from nncx.optimizers import SGD
from nncx.trainer import train, evaluate
from nncx.metrics import ClassificationMetrics
from nncx.visualizer import plot_confusion_matrix, visualize_predictions
from nncx.enums import BackendType

if __name__ == '__main__':
    do_train = False
    batch_size = 32
    
    backend_type = BackendType.CPU
    
    train_ds = FashionMNISTTrain()
    val_test_ds = FashionMNISTTest()
    _, val_ds, test_ds = val_test_ds.split_dataset(train_ratio=0, shuffle=True, test_set=True, seed=42)
    
    transform_x = [Normalize(min_val=0, max_val=255.0), Flatten()]
    transform_y = [OneHotEncode(train_ds.num_labels)]
    
    train_ds.set_transforms(transform_x, transform_y)
    val_test_ds.set_transforms(transform_x, transform_y)
    
    dl = dict()
    dl['train'] = DataLoader(train_ds, backend_type=backend_type, batch_size=batch_size, shuffle=True)
    dl['val'] = DataLoader(val_ds, backend_type=backend_type, batch_size=batch_size, shuffle=False)
    dl['test'] = DataLoader(test_ds, backend_type=backend_type, batch_size=batch_size, shuffle=False)

    first_batch = next(iter(dl['train']))
    model = Classifier(first_batch[0].shape[-1], train_ds.num_labels, backend_type)
    
    loss_fn = CrossEntropyLoss()
    
    if do_train:
        opt = SGD(model.parameters(), lr=0.005)
        train(model, loss_fn, opt, dl, 25)
        
        model.save_parameters('weights/fmnist_cls.npz')
        
    else:
        model.load_parameters('weights/fmnist_cls.npz')
    
    preds, targets = evaluate(model, loss_fn, dl)
    
    cls_metrics = ClassificationMetrics()
    
    cls_metrics.accuracy(preds, targets)
    cls_metrics.precision_recall_f1(preds, targets, num_classes=val_test_ds.num_labels)
    
    plot_confusion_matrix(preds, targets, num_classes=val_test_ds.num_labels, class_names=val_test_ds.label_names)
    
    visualize_predictions(model, dl['test'])
    
    
    
    
    
