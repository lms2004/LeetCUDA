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
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32_kernel(const int *idx, float *weight,
                                     float *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f32x4_kernel(const int *idx, float *weight,
                                       float *output, int n, int emb_size) {
  int tx = threadIdx.x * 4;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}

__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight,
                                            float *output, int n,
                                            int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + 4 * tx]) =
      LDST128BITS(weight[offset + 4 * tx]);
}

__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output,
                                     int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

__global__ void embedding_f16x8_kernel(const int *idx, half *weight,
                                       half *output, int n, int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
  output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
  output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
  output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
  output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                            half *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + 8 * tx]) =
      LDST128BITS(weight[offset + 8 * tx]);
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int N = a.size(0);                                                   \
    const int emb_size = weight.size(1);                                       \
    dim3 block(emb_size / n_elements);                                         \
    dim3 grid(N);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
  }

TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack);
}
