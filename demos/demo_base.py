import cv2
import numpy as np

from nncx.tensor import Tensor
from nncx.enums import DataType, BackendType

class BaseDemo:
    def __init__(self, model, transforms_inputs=None, window_name='Demo'):
        self.model = model
        self.transforms_inputs = transforms_inputs
        self.window_name = window_name
        self.running = True
        
    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        frame = frame.transpose(2, 0, 1)
        
        for transform in self.transforms_inputs:
            frame, _ = transform(frame, None, None)
        
        frame = np.expand_dims(frame, axis=0)
        
        data_dtype = DataType.FLOAT32 if np.issubdtype(frame.dtype, np.floating) else DataType.INT32
        
        return Tensor(frame, backend_type=self.model.backend_type, dtype=data_dtype)
    
    def postprocess(self, frame):
        return frame
    
    def run(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise Exception('Cannot open camera')
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                        
            frame_t = self.preprocess(frame)
            with Tensor.no_grad():
                preds = self.model(frame_t)
                frame = self.postprocess(preds, frame, frame_t)
            
            cv2.imshow(self.window_name, frame)
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
               self.running = False
               
        cap.release()
        cv2.destroyAllWindows() 