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
    lib.launch_elementwise_add_f32x4(a, b, c)
    print(c)
