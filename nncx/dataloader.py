import math
import numpy as np
import multiprocessing as mp
import threading
from queue import Queue
import cupy as cp

from nncx.tensor import Tensor
from nncx.utils import timeit
from nncx.enums import BackendType

class DataLoader:
    def __init__(self, dataset, backend_type, batch_size, shuffle=True, num_workers=4, max_prefetch=4) -> None:
        self.dataset = dataset
        self.backend_type = backend_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_prefetch = max_prefetch
        
        self.idxs = np.arange(len(self.dataset))
        
    def _iter_batches(self):
        for i in range(0, len(self.idxs), self.batch_size):
            batch_idx = self.idxs[i:i+self.batch_size]
            if self.num_workers > 0:
                batch = self._pool.map(self.dataset.__getitem__, batch_idx)
            else:
                batch = [self.dataset[idx] for idx in batch_idx]
                
            yield self._collate(batch)
            
    def _async_to_device(self, batch):
        inputs, targets = batch
        stream = cp.cuda.Stream(non_blocking=True)
        
        with stream:
            inputs = inputs.to(self.backend_type)
            if isinstance(targets, (list, tuple)):
                targets = tuple(t.to(self.backend_type) for t in targets)
            else:
                targets = targets.to(self.backend_type)
                
        return inputs, targets, stream
    
    def _prefetch_thread(self):
        for batch in self._iter_batches():
            if self.backend_type is BackendType.GPU:
                inputs, targets, stream = self._async_to_device(batch)
                self._queue.put((inputs, targets, stream))
            else:
                self._queue.put((*batch, None))
        
        self._queue.put(None)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
        
        self._pool = mp.Pool(self.num_workers) if self.num_workers > 0 else None
        self._queue = Queue(self.max_prefetch)
        thread = threading.Thread(target=self._prefetch_thread, daemon=True)
        thread.start()
        
        try:
            while True:
                batch = self._queue.get()
                if batch is None:
                    break
                
                inputs, targets, stream = batch
                if stream is not None:
                    stream.synchronize()
                
                yield inputs, targets
                
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
            
            
    def __len__(self):
        if self.idxs is None:
            return math.ceil(len(self.dataset)/float(self.batch_size))
        else:
            return math.ceil(len(self.idxs) / float(self.batch_size))
        
    def _collate(self, batch):    
        batch_tensors = []    
        for items in zip(*batch):
            if isinstance(items[0], (tuple, list)):
                items = tuple(Tensor.stack([t[i] for t in items]) for i in range(len(items[0])))
            else:
                items = Tensor.stack(items)
                
            batch_tensors.append(items)
        
        return tuple(batch_tensors)