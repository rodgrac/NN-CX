from nncx.backend.utils import init_backend
from nncx.datasets.wider_face import WIDERFace
from nncx.datasets import transform
from nncx.dataloader import DataLoader
from nncx.models.face_detector import FaceDetector
from nncx.losses import BCEWithLogitsLoss, SmoothL1Loss
from nncx.optimizers import SGD
from nncx import schedulers
from nncx.trainer import train, evaluate
from nncx.enums import BackendType
import nncx.visualizer as viz


class DetectionLoss:
    def __init__(self, lambda_box=10.0):
        self.bce = BCEWithLogitsLoss()
        self.smooth_l1 = SmoothL1Loss()
        self.lambda_box = lambda_box
        
    def __call__(self, preds, targets):
        cls_loss = self.bce(preds[1], targets[1])
        
        # Mask out invalid targets with no bbox
        box_loss = self.smooth_l1(preds[0], targets[0], mask=targets[1])
        
        print(f"[DETECTION LOSS] Box loss: {box_loss.detach().data:.2f}, cls loss: {cls_loss.detach().data:.2f}")
        
        return self.lambda_box * box_loss + cls_loss
        
    
if __name__ == '__main__':
    do_train = True
    input_size = 128
    batch_size = 16
    epochs = 25
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_epochs = 2
    
    backend = init_backend(BackendType.GPU)
    
    train_val_ds = WIDERFace(split='train', num_faces='single', include_negatives=True)
    test_ds = WIDERFace(split='val', num_faces='single', include_negatives=True)
    
    train_ds, val_ds = train_val_ds.split_dataset(test_set=False, seed=42)
    
    train_transforms_x = [
        transform.ResizeLetterbox(size=input_size),
        transform.RandomHorizontalFlip(p=0.5),
        transform.Normalize(min_val=0, max_val=255.0), 
        transform.Standardize(train_ds.data_mean, train_ds.data_std)
    ]
    
    val_test_transforms_x = [
        transform.ResizeLetterbox(size=input_size),
        transform.Normalize(min_val=0, max_val=255.0), 
        transform.Standardize(val_ds.data_mean, val_ds.data_std)
    ]
    
    train_ds.set_transforms(train_transforms_x)
    val_ds.set_transforms(val_test_transforms_x)
    test_ds.set_transforms(val_test_transforms_x)
    
    viz.view_image_dataset(train_ds, backend)
    
    dl = dict()
    dl['train'] = DataLoader(train_ds, backend=backend, batch_size=batch_size, shuffle=True)
    dl['val'] = DataLoader(val_ds, backend=backend, batch_size=batch_size, shuffle=False)
    dl['test'] = DataLoader(test_ds, backend=backend, batch_size=batch_size, shuffle=False)
    
    model = FaceDetector(backend, in_size=input_size)
    
    loss_fn = DetectionLoss()
    
    if do_train:
        opt = SGD(backend, model.parameters(), lr=max_lr, momentum=0.9)
        
        cosine_sched = schedulers.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
        sched = schedulers.WarmupLR(opt, cosine_sched, warmup_epochs, warmup_start_lr=min_lr)
        
        train(model, loss_fn, opt, dl, epochs, sched=sched)
        
        model.save_parameters('weights/face_detector.npz')
    else:
        model.load_parameters('weights/face_detector.npz')
        
    preds, targets = evaluate(model, loss_fn, dl)
    
   
    

    
    
    
    
    
    