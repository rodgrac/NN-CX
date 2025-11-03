import os
import random
import numpy as np
from PIL import Image

from nncx.enums import DataType, BackendType
from nncx.tensor import Tensor
from nncx.datasets.dataset import Dataset
from nncx.datasets.utils import download_extract_file
from nncx.metrics import DetectionMetrics

class WIDERFace(Dataset):
    def __init__(self, split='train', num_faces='single', include_negatives=False):
        super().__init__()
        self.name = 'WIDERFace'
        self.root = f'{self.datasets_root}/{self.name}'
        self.num_faces = num_faces
                
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

            num_annots = int(lines[i]); i += 1
            boxes = []
            
            if num_annots == 0:
                i += 1
                continue
            
            if self.num_faces == 'single' and num_annots != 1:
                i += num_annots
                continue
                        
            for _ in range(num_annots):
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
                
                if include_negatives:
                    H, W = Image.open(img_path).size[::-1]
                    nx, ny, nw, nh = self._sample_negetives(boxes, W, H)
                    self.inputs.append((img_path, (nx, ny, nw, nh)))
                    self.targets.append([(0, 0, 0, 0)])
        
        self.target_type = Dataset.TargetType.BBOX
        self.data_mean = [0.5, 0.5, 0.5]
        self.data_std = [0.5, 0.5, 0.5]
        
                
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):        
        data_item = self.inputs[idx]
        target_item = self.targets[idx]
        
        img_path = data_item if isinstance(data_item, str) else data_item[0]
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        
        # crop if negetive sample
        if isinstance(data_item, tuple):
            _, (x, y, w, h) = data_item
            img = img[y:y+h, x:x+w, :]
            label = [0.0]
        else:
            label = [1.0]
        
        x, y, w, h = target_item[0]     # For single face
      
        # Normalize
        H, W = img.shape[:2]
        bbox = self._convert_bbox_to_center_normalized(target_item[0], W, H)
        
        # HWC -> CHW before transform
        img = img.transpose(2, 0, 1)
        
        for transform in self.transforms_inputs:
            img, bbox  = transform(img, bbox, self.target_type)
            
        data_dtype = DataType.FLOAT32 if np.issubdtype(img.dtype, np.floating) else DataType.INT32
        target_dtype = DataType.FLOAT32 if np.issubdtype(bbox.dtype, np.floating) else DataType.INT32
        
        return Tensor(img, backend_type=BackendType.CPU, dtype=data_dtype, grad_en=True), \
                (Tensor(bbox, backend_type=BackendType.CPU, dtype=target_dtype), \
                Tensor(label, backend_type=BackendType.CPU, dtype=target_dtype))
    
    
    def _sample_negetives(self, boxes, W, H):
        """Sample a random crop far from any face (IoU < 0.1)."""
        iou = DetectionMetrics().IoU
        for _ in range(50):
            size = random.randint(32, min(W, H) // 2)
            x = random.randint(0, W - size)
            y = random.randint(0, H - size)
            crop = (x, y, size, size)
            crop_cnorm = self._convert_bbox_to_center_normalized(crop, W, H)
            
            # Convert to center normalized
            boxes_cnorm = []
            for box in boxes:
                boxes_cnorm.append(self._convert_bbox_to_center_normalized(box, W, H))
            
            if all(iou(box, crop_cnorm) < 0.1 for box in boxes_cnorm):
                return x, y, size, size
        
        # fallback (no safe region found)
        return 0, 0, min(W, H)//2, min(W, H)//2

    @staticmethod
    def _convert_bbox_to_center_normalized(bbox, W, H):
        x, y, w, h = bbox
        bbox_cnorm = np.array([
            (x + 0.5 * w) / W,
            (y + 0.5 * h) / H,
            w / W,
            h / H
        ], dtype=np.float32)
        
        return bbox_cnorm