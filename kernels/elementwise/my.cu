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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// kernel function
__global__ void Idtest_kernel(int *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
      a[idx] = blockDim.x;
    }
}

__global__ void elementwise_kernel(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_f32x4_kernel(float *a, float *b, float *c, int N, int* seq) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  seq[idx] = blockDim.x;
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

// launch kernel function
void launch_Idtest_kernel(torch::Tensor a) {
  int N = a.numel();
  int block_size = 256;
  int grid_size = N / 256;
  Idtest_kernel<<<grid_size, block_size>>>(
    reinterpret_cast<int *>(a.data_ptr()),
    N);
}

void launch_elementwise_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  int N = a.numel();
  int block_size = 256;
  int grid_size = N / 256;
  elementwise_kernel<<<grid_size, block_size>>>(
    reinterpret_cast<float *>(a.data_ptr()),
    reinterpret_cast<float *>(b.data_ptr()),
    reinterpret_cast<float *>(c.data_ptr()),
    N
  );
}

void launch_elementwise_f32x4_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor seq) {
  int N = a.numel();
  int block_size = 256 / 4;
  int grid_size = N / 256;
  elementwise_f32x4_kernel<<<grid_size, block_size>>>(
    reinterpret_cast<float *>(a.data_ptr()),
    reinterpret_cast<float *>(b.data_ptr()),
    reinterpret_cast<float *>(c.data_ptr()),
    N,
    reinterpret_cast<int *>(seq.data_ptr())
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_Idtest_kernel", &launch_Idtest_kernel, "launch_Idtest_kernel (CUDA)");
  m.def("launch_elementwise_kernel", &launch_elementwise_kernel, "launch_elementwise_kernel (CUDA)");
  m.def("launch_elementwise_f32x4_kernel", &launch_elementwise_f32x4_kernel, "launch_elementwise_f32x4_kernel (CUDA)");
}
