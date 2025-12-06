import numpy as np

from nncx.utils import sigmoid

class ClassificationMetrics:
    def accuracy(self, preds, targets):
        preds, targets = preds.get(), targets.get()
        
        if preds.ndim != 1:     # Sample
            preds = np.argmax(preds, axis=-1)
            
        if targets.ndim != 1:     # Sample
            targets = np.argmax(targets, axis=-1)
            
        acc = np.mean(preds == targets)
        
        print(f"[METRICS] Accuracy: {acc}")
            
        return acc
    
    
    def precision_recall_f1(self, preds, targets, num_classes, reduce_weighted_mean=True):
        preds, targets = preds.get(), targets.get()
        
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
    
    
class DetectionMetrics:
    def __init__(self, iou_thresh=0.5, conf_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.reset()
    
    def IoU(self, a, b, center_format=True):  # (cx, cy, w, h)
        if center_format:   
            a_x1 = a[0] - a[2] / 2
            a_y1 = a[1] - a[3] / 2
            a_x2 = a[0] + a[2] / 2
            a_y2 = a[1] + a[3] / 2
            
            b_x1 = b[0] - b[2] / 2
            b_y1 = b[1] - b[3] / 2
            b_x2 = b[0] + b[2] / 2
            b_y2 = b[1] + b[3] / 2
            
            # (x1, y1, x2, y2)
            a, b = (a_x1, a_y1, a_x2, a_y2), (b_x1, b_y1, b_x2, b_y2)
        
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1)
        areaB = max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
        union = areaA + areaB - inter + 1e-9
        
        return inter / union
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.ious = []
        
    def compute(self, preds, targets):        
        for pred in preds:
            pred_bbox, conf = tuple(p.get() for p in pred)
            if sigmoid(conf) < self.conf_thresh:
                continue
            
            matched = False
            for target in targets:
                gt_bbox, _ = tuple(t.get() for t in target)
                iou = self.IoU(pred_bbox, gt_bbox)
                if iou >= self.iou_thresh:
                    self.tp += 1
                    self.ious.append(iou)
                    matched = True
                    break
            if not matched:
                self.fp += 1
                
        self.fn += max(0, len(targets) - self.tp)
        
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        mean_iou = np.mean(self.ious) if self.ious else 0.0
        
        print(f"[Metrics] Precision: {precision:.3f}, Recall: {recall:.3f}, "
              f"F1: {f1:.3f}, Mean IoU: {mean_iou:.3f}")
        
        