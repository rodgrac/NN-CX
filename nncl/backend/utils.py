from nncl.backend.backend_cpu import CPUBackend
from nncl.backend.backend_gpu import GPUBackend

from nncl.enums import BackendType


def init_backend(backend_type):    
    if backend_type == BackendType.CPU:
        return CPUBackend()
    elif backend_type == BackendType.GPU:
        return GPUBackend()
    else:
        raise NotImplementedError

    
def get_backend_type(backend):
    if isinstance(backend, CPUBackend):
        return BackendType.CPU
    elif isinstance(backend, GPUBackend):
        return BackendType.GPU
    else:
        raise ValueError