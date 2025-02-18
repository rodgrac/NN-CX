from enum import Enum, auto

class DataType(Enum):
    FLOAT32 = auto()
    INT32 = auto()
    
    def __str__(self):
        return self.name
    
class BackendType(Enum):
    CPU = auto()
    GPU = auto()
    
    def __str__(self):
        return self.name