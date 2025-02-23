import numpy as np


class ClassificationMetrics:
    def accuracy(self, preds, targets):
        preds, targets = np.array(preds), np.array(targets)
        
        if preds.ndim != 1:     # Sample
            preds = np.argmax(preds, axis=-1)
            
        if targets.ndim != 1:     # Sample
            targets = np.argmax(targets, axis=-1)
            
        acc = np.mean(preds == targets)
        
        print(f"[METRICS] Accuracy: {acc}")
            
        return acc
    
    
    def precision_recall_f1(self, preds, targets, num_classes, reduce_weighted_mean=True):
        preds, targets = np.array(preds), np.array(targets)
        
        if preds.ndim != 1:     # Sample
            preds = np.argmax(preds, axis=-1)
            
        if targets.ndim != 1:     # Sample
            targets = np.argmax(targets, axis=-1)
            
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        
        for cls in range(num_classes):
            tp = np.sum((cls == preds) & (cls == targets))
            fp = np.sum((cls == preds) & (cls != targets))
            fn = np.sum((cls != preds) & (cls == targets))
            
            precision[cls] = tp / (tp + fp) if tp + fp > 0 else 0
            recall[cls] = tp / (tp + fn) if tp + fn > 0 else 0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        
        if reduce_weighted_mean:
            cls_cnt = np.array([np.sum(targets == i) for i in range(num_classes)])
            cls_freq = cls_cnt / np.sum(cls_cnt)
            
            precision = np.sum(cls_freq * precision)
            recall = np.sum(cls_freq * recall)
            f1 = np.sum(cls_freq * f1)
            
        print(f"[METRICS] F1 score: {f1}")
        print(f"[METRICS] Precision: {precision}")
        print(f"[METRICS] Recall: {recall}")
        
        return precision, recall, f1