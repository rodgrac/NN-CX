import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def visualize_predictions(preds, targets, title='', xlabel='Targets', ylabel='Predictions'):
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
    