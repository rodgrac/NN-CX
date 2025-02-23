import numpy as np
import matplotlib.pyplot as plt
from typing import List


def visualize_predictions(preds, targets, title='', xlabel='Targets', ylabel='Predictions'):
    if isinstance(preds, List):
        preds = np.array(preds)
        targets = np.array(targets)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, preds, color='blue', alpha=0.6)
    
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    