import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def view_image_dataset(dataset, grid_size=(5, 5)):
    fig, axs = plt.subplots(*grid_size, figsize=(8, 8))
    axs = axs.flatten()
    
    idxs = random.sample(range(len(dataset)), k=grid_size[0] * grid_size[1])
        
    for i in range(len(idxs)):
        img = dataset.inputs[idxs[i]]
        if img.ndim == 1:   # Flattened
            img = img.reshape(dataset.image_size)
        else:
            img = img.transpose(1, 2, 0)    # HWC
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(dataset.label_names[dataset.targets[idxs[i]]], fontsize=8)
    
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
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(inputs_s[i].reshape(dataloader.dataset.image_size))
        plt.axis('off')
        plt.title(f"Pred: {dataloader.dataset.label_names[int(preds_s[i])]}\nTrue: {dataloader.dataset.label_names[int(targets_s[i])]}", 
                  fontsize=10, color='green' if preds_s[i] == targets_s[i] else 'red')
    
    plt.tight_layout()
    plt.show()