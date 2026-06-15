import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List

from nncx.datasets.dataset import Dataset
from nncx.utils import sigmoid

def view_image_dataset(dataset: Dataset, grid_size=(5, 5)):
    fig, axs = plt.subplots(*grid_size, figsize=(8, 8))
    axs = axs.flatten()
    
    idxs = random.sample(range(len(dataset)), k=grid_size[0] * grid_size[1])
        
    for i in range(len(idxs)):
        input, target = dataset[idxs[i]]
        img = input.get()
        if img.ndim == 1:   # Flattened
            img = img.reshape(dataset.image_size)
        elif img.shape[0] == 3: # CHW
            img = img.transpose(1, 2, 0)    # HWC
            
        axs[i].imshow(img)
        axs[i].axis('off')
        if dataset.target_type == Dataset.TargetType.ONE_HOT:
            axs[i].set_title(dataset.label_names[target.get()], fontsize=8)
        elif dataset.target_type == Dataset.TargetType.BBOX:
            target = target[0] if isinstance(target, tuple) else target
            xc, yc, w, h = target.get()
            H, W = img.shape[:2]
            x = (xc - w/2) * W
            y = (yc - h/2) * H
            w *= W; h *= H
            axs[i].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
        
    plt.tight_layout()
    plt.show()


def plot_predictions_targets(preds, targets, title='', xlabel='Targets', ylabel='Predictions'):
    if isinstance(preds, List):
        preds = np.array(preds)
        targets = np.array(targets)
    
    
    print('[Viz] Plotting predictions vs targets...')
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, preds, color='blue', alpha=0.6)
    
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    
    
def plot_confusion_matrix(preds, targets, num_classes, class_names):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    preds, targets = np.array(preds), np.array(targets)
    if preds.ndim != 1:     # Sample
        preds = np.argmax(preds, axis=-1)
        
    if targets.ndim != 1:     # Sample
        targets = np.argmax(targets, axis=-1)
    
    for t, p in zip(targets, preds):
        cm[t, p] += 1
        
    plt.figure(figsize=(10, 8))
    print('[Viz] Plotting confusion matrix...')
    sns.heatmap(cm, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.title('Confusion Matrix') 
    plt.show()   
    
    
def visualize_predictions(model, dataloader, num_samples=9):
    print('[Viz] Visualizing predictions...')
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # sample a random batch
    num_workers_tmp = dataloader.num_workers
    dataloader.num_workers = 0
    batch_idx = random.randint(0, len(dataloader) - 1)
    for i, batch in enumerate(dataloader):
        if i == batch_idx:
            inputs, targets = batch
            break
        
    sample_idxs = np.random.choice(inputs.shape[0], num_samples, replace=False)
    
    inputs_s = inputs[sample_idxs]
    targets_s =  tuple(t[sample_idxs] for t in targets)
    
    preds_s = model.predict(inputs_s)
    
    inputs_s, preds_s, targets_s = inputs_s.get(), tuple(p.get() for p in preds_s), tuple(t.get() for t in targets_s)
    
    # if targets_s.ndim != 1:
    #     targets_s = np.argmax(targets_s, axis=-1)
    
    pred_boxes, pred_confs = preds_s
    gt_boxes, _ = targets_s
    
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i in range(num_samples):
        img = inputs_s[i]
        if img.ndim != 3:
           img = img.reshape(dataloader.dataset.image_size)
        
        # Invert any transforms
        for transform in dataloader.dataset.transforms_inputs[::-1]:    # Reverse transform order
            if hasattr(transform, 'invert') and  callable(getattr(transform, 'invert')):
                img = transform.invert(img)
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img.transpose(1, 2, 0))   # HWC
        plt.axis('off')
        
        
        H, W = img.shape[-2], img.shape[-1]
        # Predicted boxes
        conf = sigmoid(pred_confs[i])
        if conf < 0.5:
            continue
        cx, cy, bw, bh = pred_boxes[i].flatten()
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), bw * W, bh * H,
            fill=False, color='lime', lw=2
        ))
        plt.text(x1, y1, f"{float(conf):.2f}", color='yellow', fontsize=8)

        # Ground truth boxes
        cx, cy, bw, bh = gt_boxes[i].flatten()
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), bw * W, bh * H,
            fill=False, color='red', lw=1.5, linestyle='--'
        ))

        plt.title(f"Conf ≥ {0.5}", fontsize=9)
        
        #FIXME: cls disabled. COver both cls and detect in this method
        # plt.title(f"Pred: {dataloader.dataset.label_names[int(preds_s[i])]}\nTrue: {dataloader.dataset.label_names[int(targets_s[i])]}", 
        #           fontsize=10, color='green' if preds_s[i] == targets_s[i] else 'red')
    
    plt.tight_layout()
    plt.show()
    
    # Revert num workers
    dataloader.num_workers = num_workers_tmp