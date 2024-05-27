import pytest
import numpy as np

from nncl.tensor import Tensor

def test_to_dtype():
    t = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32).rand(-10, 10)
    
    expected = t.data.astype(np.int32)
    
    t.to_dtype(Tensor.INT32)
    
    assert np.array_equal(t.data, expected)

def test_tensor_add():
    t1 = Tensor(shape=(10, 5), dtype=Tensor.INT32).rand(-10, 10)
    t2 = Tensor(shape=(10, 5), dtype=Tensor.INT32).rand(-10, 10)
    
    t = t1 + t2
    
    expected = t1.data + t2.data
    assert np.array_equal(t.data, expected)
    
def test_clone():
    t1 = Tensor(shape=(10, 5), dtype=Tensor.FLOAT32).rand(-10, 10)
    t2 = t1.clone()
    
    expected = t1.data
    assert np.array_equal(t2.data, expected)
    

    
