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

__global__ void elementwise_f32_kernel(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_f32x4_kernel(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
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


#define ADD_ELEMENTWISE(packed_type, element_type, n_elements)                            \
  void launch_elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b, torch::Tensor c) { \
    int N = a.numel();                                                                    \
    int block_size = 256;                                                                  \
    int grid_size = (N + 255) / 256;                                                      \
    elementwise_##packed_type##_kernel<<<grid_size, block_size>>>(                        \
      reinterpret_cast<element_type *>(a.data_ptr()),                                     \
      reinterpret_cast<element_type *>(b.data_ptr()),                                     \
      reinterpret_cast<element_type *>(c.data_ptr()),                                     \
      N                                                                                   \
    );                                                                                    \
  }                                                                                       \



ADD_ELEMENTWISE(f32x4, float, 4);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_elementwise_add_f32x4", &launch_elementwise_add_f32x4, "launch_elementwise_add_f32x4 (CUDA)");
}
