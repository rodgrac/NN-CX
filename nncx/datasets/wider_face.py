import os
import numpy as np
from PIL import Image

from nncx.enums import DataType
from nncx.tensor import Tensor
from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_extract_file

class WIDERFace(Dataset):
    def __init__(self, split='train', pick_face='largest'):
        super().__init__()
        self.name = 'WIDERFace'
        self.root = f'{self.datasets_root}/{self.name}'
        self.pick_face = pick_face
                
        self.download_url = {
            'train': 'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=587b0d62-730f-477d-91c4-4a59d5b595c1&at=AKSUxGOTBSnQlu6z_1nvXLHuOqV8%3A1761428739562',
            'val': 'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=767bd9d8-b540-40a2-a801-e8c7f7bb23f3&at=AKSUxGPdyU1ntkBUIaFjCWHUKt9J%3A1761428824335',
            'labels': 'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
        }
        
        download_extract_file(self.download_url[split], self.root, file_name=f'WIDER_{split}.zip')
        download_extract_file(self.download_url['labels'], self.root)

        self.inputs = []
        self.targets = []
        
        with open(os.path.join(self.root, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt'), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            img_path = lines[i]; i += 1
            if "/" not in img_path:
                raise ValueError(f"Expected image path, got: {img_path!r} at line {i}")

            num_faces = int(lines[i]); i += 1
            boxes = []
            
            if num_faces == 0:
                i += 1
                continue
            
            for _ in range(num_faces):
                attr_vals = list(map(int, lines[i].split()))
                i += 1
                x, y, w, h = attr_vals[:4]
                blur, expr, illum, invalid, occ, pose = attr_vals[4:10]
                if invalid == 0 and w > 0 and h > 0:
                    boxes.append((x, y, w, h))
                    
            img_path = f"{self.root}/WIDER_{split}/images/{img_path}"
            if boxes and os.path.exists(img_path):
                self.inputs.append(img_path)
                self.targets.append(boxes)
        
        self.target_type = Dataset.TargetType.BBOX
        
                
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, key):
        idx, backend = key
        
        data_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        data_item = np.array(Image.open(data_item).convert('RGB'), dtype=np.uint8)
        H, W = data_item.shape[:2]
        
        if self.pick_face == 'largest':    # largest face
            x, y, w, h = max(target_item, key=lambda x: x[2] * x[3])
        else:                                  
            raise NotImplementedError
        
        # Normalize
        target_item = np.array([
            (x + 0.5 * w) / W,
            (y + 0.5 * h) / H,
            w / W,
            h / H
        ], dtype=np.float32)
        
        for transform in self.transforms_inputs:
            data_item = transform(data_item)
            
        for transform in self.transforms_targets:
            target_item = transform(target_item)
            
        data_dtype = DataType.FLOAT32 if np.issubdtype(data_item.dtype, np.floating) else DataType.INT32
        target_dtype = DataType.FLOAT32 if np.issubdtype(target_item.dtype, np.floating) else DataType.INT32
        
        return Tensor(data_item, backend=backend, dtype=data_dtype, grad_en=True), \
                Tensor(target_item, backend=backend, dtype=target_dtype)
        