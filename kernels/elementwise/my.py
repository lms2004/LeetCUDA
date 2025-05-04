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
    N = 1024
    a = torch.ones(N, dtype=torch.float32,device="cuda")
    b = torch.ones(N, dtype=torch.float32,device="cuda")
    c = torch.zeros(N, dtype=torch.float32,device="cuda")
    seq = torch.zeros(N, dtype=torch.int32,device="cuda")
    lib.launch_elementwise_f32x4_kernel(a, b, c, seq)
    print(c)
