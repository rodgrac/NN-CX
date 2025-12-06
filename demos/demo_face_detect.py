import cv2
import numpy as np

from nncx.enums import BackendType
from nncx.models.face_detector import FaceDetector
from nncx.datasets import transform
from nncx.utils import sigmoid
from demos.demo_base import BaseDemo

class FaceDetectDemo(BaseDemo):
    def __init__(self, model, transforms_inputs=None, window_name='Demo'):
        super().__init__(model, transforms_inputs, window_name)
        
    def postprocess(self, preds, frame_org, frame_pp):
        bbox, conf = preds
        
        cur_size = frame_pp.shape[-1]
        cx, cy, bw, bh = bbox.get().flatten() * cur_size
        conf = sigmoid(conf.get()[0])
        
        # Compute scaling & padding applied during letterboxing
        scale = cur_size / max(frame_org.shape[-2], frame_org.shape[-1])
        new_h, new_w = int(scale * frame_org.shape[-2]), int(scale * frame_org.shape[-1])
        pad_top = (cur_size - new_h) // 2
        pad_left = (cur_size - new_w) // 2
        
        # Undo padding
        cx -= pad_left
        cy -= pad_top

        # Undo scaling
        cx /= scale
        cy /= scale
        bw /= scale
        bh /= scale
        
        # Clip to original frame
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(frame_org.shape[-1], cx + bw / 2))
        y2 = int(min(frame_org.shape[-2], cy + bh / 2))
        
        cv2.rectangle(frame_org, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(
            frame_org,
            f"{float(conf):.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
        
        return frame_org
    

if __name__ == '__main__':
    input_size = 128
    data_mean = [0.5, 0.5, 0.5]
    data_std = [0.5, 0.5, 0.5]
    
    backend_type = BackendType.GPU
    
    model = FaceDetector(backend_type, in_size=128)
    model.load_parameters(f'weights/{model.__class__.__name__}/best_model.npz')

        
    transforms_inputs = [
        transform.ResizeLetterbox(size=input_size),
        transform.Normalize(min_val=0, max_val=255.0), 
        transform.Standardize(data_mean, data_std)
    ]
        
    demo = FaceDetectDemo(model, transforms_inputs=transforms_inputs)
    demo.run()