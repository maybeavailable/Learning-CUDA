#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
#define BLOCK_SIZE 256
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void nv_trace(T* input, size_t diag_len, size_t cols, T* value) {
  __shared__ T sdata[BLOCK_SIZE / 32];
  
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  int lane = tid % warpSize;
  int warpId = tid / warpSize;
  
  T local_sum = T(0);
  for (size_t i = idx; i < diag_len; i += stride) {
    local_sum += input[i * cols + i];
  }
  
  local_sum = warpReduceSum(local_sum);
  
  if (lane == 0) {
    sdata[warpId] = local_sum;
  }
  __syncthreads();
  
  if (warpId == 0) {
    local_sum = (tid < blockDim.x / warpSize) ? sdata[lane] : T(0);
    local_sum = warpReduceSum(local_sum);
    
    if (tid == 0) {
      atomicAdd(value, local_sum);
    }
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  size_t diag_len = (rows > cols ? cols : rows);
  
  T* input = nullptr;
  cudaMalloc(&input, h_input.size() * sizeof(T));
  
  T* value = nullptr;
  cudaMalloc(&value, sizeof(T));
  cudaMemset(value, 0, sizeof(T));
  
  cudaMemcpy(input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);
  
  const int blockSize = BLOCK_SIZE;
  int gridSize = (diag_len + blockSize - 1) / blockSize;
  
  nv_trace<T><<<gridSize, blockSize>>>(input, diag_len, cols, value);
  
  T result;
  cudaMemcpy(&result, value, sizeof(T), cudaMemcpyDeviceToHost);
  
  cudaFree(input);
  cudaFree(value);
  
  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

#define FA_BLOCK_M 64
#define FA_BLOCK_N 64

template <typename T>
__device__ __forceinline__ float toFloat(T val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float toFloat(half val) {
  return __half2float(val);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float val) {
  return static_cast<T>(val);
}

template <>
__device__ __forceinline__ half fromFloat(float val) {
  return __float2half(val);
}

template <typename T>
__global__ void flashAttentionKernel(
    const T*  Q,
    const T*  K,
    const T*  V,
    T*  O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    float scale, bool is_causal) {
  
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int q_tile_idx = blockIdx.x;
  
  int kv_head_idx = head_idx / (query_heads / kv_heads);
  
  int q_start = q_tile_idx * FA_BLOCK_M;
  if (q_start >= tgt_seq_len) return;
  
  int q_end = min(q_start + FA_BLOCK_M, tgt_seq_len);
  int q_len = q_end - q_start;
  
  int q_offset = batch_idx * tgt_seq_len * query_heads * head_dim + head_idx * head_dim;
  int k_offset = batch_idx * src_seq_len * kv_heads * head_dim + kv_head_idx * head_dim;
  int v_offset = k_offset;
  int o_offset = q_offset;
  
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  
  extern __shared__ float smem[];
  float* s_rowmax = smem;
  float* s_rowsum = smem + FA_BLOCK_M;
  float* s_O = smem + 2 * FA_BLOCK_M;
  
  for (int i = tid; i < FA_BLOCK_M; i += num_threads) {
    s_rowmax[i] = -INFINITY;
    s_rowsum[i] = 0.0f;
  }
  for (int i = tid; i < FA_BLOCK_M * head_dim; i += num_threads) {
    s_O[i] = 0.0f;
  }
  __syncthreads();
  
  for (int kv_start = 0; kv_start < src_seq_len; kv_start += FA_BLOCK_N) {
    int kv_tile_end = min(kv_start + FA_BLOCK_N, src_seq_len);
    int kv_len = kv_tile_end - kv_start;
    
    for (int m = tid; m < q_len; m += num_threads) {
      int q_pos = q_start + m;
      float old_max = s_rowmax[m];
      float old_sum = s_rowsum[m];
      float new_max = old_max;
      
      for (int n = 0; n < kv_len; n++) {
        int kv_pos = kv_start + n;
        
        if (is_causal && kv_pos > q_pos) {
          continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          float q_val = toFloat(Q[q_offset + q_pos * query_heads * head_dim + d]);
          float k_val = toFloat(K[k_offset + kv_pos * kv_heads * head_dim + d]);
          score += q_val * k_val;
        }
        score *= scale;
        new_max = fmaxf(new_max, score);
      }
      
      float exp_diff = expf(old_max - new_max);
      for (int d = 0; d < head_dim; d++) {
        s_O[m * head_dim + d] *= exp_diff;
      }
      float new_sum = old_sum * exp_diff;
      
      for (int n = 0; n < kv_len; n++) {
        int kv_pos = kv_start + n;
        
        if (is_causal && kv_pos > q_pos) {
          continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          float q_val = toFloat(Q[q_offset + q_pos * query_heads * head_dim + d]);
          float k_val = toFloat(K[k_offset + kv_pos * kv_heads * head_dim + d]);
          score += q_val * k_val;
        }
        score *= scale;
        
        float attn_weight = expf(score - new_max);
        new_sum += attn_weight;
        
        for (int d = 0; d < head_dim; d++) {
          float v_val = toFloat(V[v_offset + kv_pos * kv_heads * head_dim + d]);
          s_O[m * head_dim + d] += attn_weight * v_val;
        }
      }
      
      s_rowmax[m] = new_max;
      s_rowsum[m] = new_sum;
    }
    __syncthreads();
  }
  
  for (int m = tid; m < q_len; m += num_threads) {
    int q_pos = q_start + m;
    float inv_sum = (s_rowsum[m] > 0.0f) ? (1.0f / s_rowsum[m]) : 0.0f;
    
    for (int d = 0; d < head_dim; d++) {
      O[o_offset + q_pos * query_heads * head_dim + d] = fromFloat<T>(s_O[m * head_dim + d] * inv_sum);
    }
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  
  size_t q_size = h_q.size() * sizeof(T);
  size_t k_size = h_k.size() * sizeof(T);
  size_t v_size = h_v.size() * sizeof(T);
  size_t o_size = h_o.size() * sizeof(T);
  
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, q_size);
  cudaMalloc(&d_k, k_size);
  cudaMalloc(&d_v, v_size);
  cudaMalloc(&d_o, o_size);
  
  cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);
  
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  
  int num_q_tiles = (target_seq_len + FA_BLOCK_M - 1) / FA_BLOCK_M;
  
  dim3 grid(num_q_tiles, query_heads, batch_size);
  dim3 block(128);
  
  size_t smem_size = (2 * FA_BLOCK_M + FA_BLOCK_M * head_dim) * sizeof(float);
  
  flashAttentionKernel<T><<<grid, block, smem_size>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      scale, is_causal);
  
  cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
