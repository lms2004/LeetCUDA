import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="my", 
    sources=["my.cu"]
)


if __name__ == "__main__":
    # Create a tensor on the GPU
    N = 1024
    a = torch.zeros(N, dtype=torch.int32,device='cuda')
    lib.launch_Idtest_kernel(a)
    
    print(a)
