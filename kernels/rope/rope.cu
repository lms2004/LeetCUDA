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

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

/*
  x: embedding
  out: embedding + rope position encoding
  seq_len: sequence length
  N: hidden_size / 2 (half of embedding size)
*/
__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的全局索引

  // 获取当前处理的复数分量 (实部x1, 虚部x2)
  float x1 = x[idx * 2];     // 实部
  float x2 = x[idx * 2 + 1]; // 虚部

  int token_pos = idx / N; // 当前token在序列中的位置 (0 到 seq_len-1)
  int token_idx = idx % N; // 当前复数分量在token内的维度索引 (0 到 N-1)

  // █ 关键公式解释 (原始RoPE公式) █
  // 旋转频率因子：θ_j = 10000^{-2j/d} 
  // 其中 d = 2N (原始embedding大小), j = token_idx
  // 代码等效公式：exp_v = 10000^{-token_idx/N}
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));

  // █ 旋转角度计算 █
  // 角度 = 位置 * 频率因子：mθ_j (m = token_pos)
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);

  // █ 复数旋转操作 (欧拉公式) █
  // [ out1 ]   [ cos(mθ_j)  -sin(mθ_j) ] [ x1 ]
  // [ out2 ] = [ sin(mθ_j)   cos(mθ_j) ] [ x2 ]
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  
  out[idx * 2] = out1;
  out[idx * 2 + 1] = out2;
}

// another index method of rope.
__global__ void rope_f32_v2_kernel(float *x, float *out, int seq_len, int N) {
  int token_pos = blockIdx.x;   // 使用blockIdx.x直接作为token位置
  int tid = threadIdx.x;        // 使用threadIdx.x作为token内分量的索引

  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];
  float exp_v = 1.0f / powf(theta, 2 * tid / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + tid * 2] = out1;
  out[token_pos * N * 2 + tid * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 一次性读取4个float（两个复数分量）
  float4 x_v = FLOAT4(x[idx * 4]);
  
  int token_pos = idx / N;
  int token_idx = idx % N;
  
  // 计算两个复数分量对应的旋转频率因子
  float exp_f_v = 1.0f / powf(theta, 2 * token_idx * 2 / (N * 4.0f));
  float exp_s_v = 1.0f / powf(theta, 2 * (token_idx * 2 + 1) / (N * 4.0f));
  
  // 计算两个旋转角度
  float sin_f_v = sinf(token_pos * exp_f_v);
  float cos_f_v = cosf(token_pos * exp_f_v);
  float sin_s_v = sinf(token_pos * exp_s_v);
  float cos_s_v = cosf(token_pos * exp_s_v);
  
  // 分别旋转两个复数分量
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;  // 第一个复数分量
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;  // 第二个复数分量
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
  
  // 一次性写回4个float
  FLOAT4(out[idx * 4]) = out_v;
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

void rope_f32(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 2);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                   seq_len, N);
}

// 输入： embedding 
// 输出：添加 rope 位置编码 embedding
void rope_f32_v2(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 2);
  dim3 grid(seq_len);
  dim3 block(N);
  rope_f32_v2_kernel<<<grid, block>>>(x.data_ptr<float>(),
                                      out.data_ptr<float>(), seq_len, N);
}

void rope_f32x4_pack(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 4);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32x4_pack_kernel<<<grid, block>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(), seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rope_f32)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32_v2)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32x4_pack)
}
