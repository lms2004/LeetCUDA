#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// kernel function
__global__ void Idtest_kernel(int *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
      a[idx] = blockDim.x;
    }
}


// launch kernel function
void launch_Idtest_kernel(torch::Tensor a) {
  int N = a.numel();
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;
  Idtest_kernel<<<grid_size, block_size>>>(
    reinterpret_cast<int *>(a.data_ptr()),
    N);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_Idtest_kernel", &launch_Idtest_kernel, "launch_Idtest_kernel (CUDA)");
}
