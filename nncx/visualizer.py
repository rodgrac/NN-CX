import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List

from nncx.datasets.dataset import Dataset

def view_image_dataset(dataset: Dataset, backend, grid_size=(5, 5)):
    fig, axs = plt.subplots(*grid_size, figsize=(8, 8))
    axs = axs.flatten()
    
    idxs = random.sample(range(len(dataset)), k=grid_size[0] * grid_size[1])
        
    for i in range(len(idxs)):
        input, target = dataset[(idxs[i], backend)]
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
    
    inputs, targets = next(iter(dataloader))
    idxs = np.random.choice(inputs.shape[0], num_samples, replace=False)
    
    inputs_s, targets_s = inputs[idxs], targets[idxs]
    
    preds_s = model.predict(inputs_s)
    
    inputs_s, preds_s, targets_s = inputs_s.get(), preds_s.get(), targets_s.get()
    
    if targets_s.ndim != 1:
        targets_s = np.argmax(targets_s, axis=-1)
    
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i, idx in enumerate(idxs):
        img = inputs_s[i].reshape(dataloader.dataset.image_size)
        
        # Invert any transforms
        for transform in dataloader.dataset.transforms_inputs[::-1]:    # Reverse transform order
            if hasattr(transform, 'invert') and  callable(getattr(transform, 'invert')):
                img = transform.invert(img)
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img.transpose(1, 2, 0))   # HWC
        plt.axis('off')
        plt.title(f"Pred: {dataloader.dataset.label_names[int(preds_s[i])]}\nTrue: {dataloader.dataset.label_names[int(targets_s[i])]}", 
                  fontsize=10, color='green' if preds_s[i] == targets_s[i] else 'red')
    
    plt.tight_layout()
    plt.show()