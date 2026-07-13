#include "flashattention.h"
#include "fa_autotune.h"
#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>
namespace transformer {

static constexpr int WARP_SIZE_FA = 32;
template <class T> __device__ __forceinline__ T to_elem(float x);
template <> __device__ __forceinline__ half to_elem<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <class T>
__device__ __forceinline__ void
ptx_mma_m16n8k16(float &d0, float &d1, float &d2, float &d3, uint32_t a0,
                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                 uint32_t b1, float c0, float c1, float c2, float c3) {
  if constexpr (::std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  }
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1,
                                            uint32_t &r2, uint32_t &r3,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                                  const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// Plain (non-transposed) x2 variant — required for the K (B-operand) load.
__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void cp_async_16(void *smem_dst,
                                            const void *gmem_src, bool pred) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  int src_size = pred ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_int),
      "l"(gmem_src), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ============================================================================
// Kernel
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_ptx_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE, const int seq_len,
    const int num_q_heads, const int num_kv_heads, const float scale) {
  const int bh_idx = blockIdx.y;            // batch * head index
  const int q_start = blockIdx.x * BLOCK_M; // first query row for this block
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;

  if (q_start >= seq_len)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int P_STRIDE = BLOCK_N + SMEM_PAD;

  extern __shared__ char smem_raw[];
  T *smem_q = reinterpret_cast<T *>(smem_raw);
  T *smem_k = smem_q + BLOCK_M * Q_STRIDE;
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;
  T *smem_p = smem_k; // alias onto K (see note above): saves 9 KB, +1 block/SM
  float *smem_partial_max =
      reinterpret_cast<float *>(smem_v + BLOCK_N * KV_STRIDE);
  float *smem_partial_sum = smem_partial_max + 2 * BLOCK_M;
  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);
  const size_t q_off = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
  const size_t kv_off =
      static_cast<size_t>(b * num_kv_heads + h_kv) * seq_len * D_HEAD;
  const T *Q_head = Q + q_off;
  const T *K_head = K + kv_off;
  const T *V_head = V + kv_off;
  T *O_head = O + q_off;
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val =
          (g < seq_len)
              ? reinterpret_cast<const uint4 *>(Q_head + g * D_HEAD)[col]
              : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_q + row * Q_STRIDE)[col] = val;
    }
  }
  __syncthreads();
  constexpr int QK_TILES_PER_WARP =
      BLOCK_N / 16; // N-cols/half ÷ 8; = 4 at BLOCK_N=64
  constexpr int PV_TILES_PER_WARP =
      D_HEAD / 16; // D-cols/half ÷ 8; = 4 at D=64, 8 at D=128
  constexpr int TILES_K = D_HEAD / 16;   // k-tiles for Q*K^T
  constexpr int TILES_BN = BLOCK_N / 16; // k-tiles for P*V

  const int warp_pair = warp_id / 2; // which 16-row tile (0-3)
  const int warp_half = warp_id % 2; // which half of N or D
  const int mi = warp_pair;          // m-tile index
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;
  float o_acc[PV_TILES_PER_WARP][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_len - kv_start);
    {
      constexpr int VEC_COLS = D_HEAD / 8;
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        bool valid = (g < seq_len) && (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(K_head + g * D_HEAD) + col,
                    valid);
      }
      cp_async_commit_group(); // commit K as its own group
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        bool valid = (g < seq_len) && (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(V_head + g * D_HEAD) + col,
                    valid);
      }
      cp_async_commit_group(); 
      cp_async_wait_group<1>();
    }
    __syncthreads();
    float s_acc[QK_TILES_PER_WARP][4];
    {
#pragma unroll
      for (int ni_local = 0; ni_local < QK_TILES_PER_WARP; ni_local++) {
        int ni = warp_half * QK_TILES_PER_WARP + ni_local;

        s_acc[ni_local][0] = 0.0f;
        s_acc[ni_local][1] = 0.0f;
        s_acc[ni_local][2] = 0.0f;
        s_acc[ni_local][3] = 0.0f;

#pragma unroll
        for (int ki = 0; ki < TILES_K; ki++) {
          // Load Q tile: A operand (row-major, m16k16)
          uint32_t a0, a1, a2, a3;
          {
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_q + (mi * 16 + row) * Q_STRIDE + ki * 16 + col);
          }
          uint32_t b0, b1;
          {
            int k_row = lane_id % 8;
            int mat = (lane_id / 8) % 2;
            ldmatrix_x2(b0, b1,
                        smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 +
                            mat * 8);
          }

          ptx_mma_m16n8k16<T>(
              s_acc[ni_local][0], s_acc[ni_local][1], s_acc[ni_local][2],
              s_acc[ni_local][3], a0, a1, a2, a3, b0, b1, s_acc[ni_local][0],
              s_acc[ni_local][1], s_acc[ni_local][2], s_acc[ni_local][3]);
        }

        // Apply scale and causal mask directly in registers
        int s_col0 = ni * 8 + (lane_id % 4) * 2;
        int s_col1 = s_col0 + 1;

#pragma unroll
        for (int i = 0; i < 4; i++)
          s_acc[ni_local][i] *= scale;

        if (CAUSAL) {
          if (kv_start + s_col0 > q_start + global_row0)
            s_acc[ni_local][0] = -FLT_MAX;
          if (kv_start + s_col1 > q_start + global_row0)
            s_acc[ni_local][1] = -FLT_MAX;
          if (kv_start + s_col0 > q_start + global_row1)
            s_acc[ni_local][2] = -FLT_MAX;
          if (kv_start + s_col1 > q_start + global_row1)
            s_acc[ni_local][3] = -FLT_MAX;
        }
        if (s_col0 >= kv_count) {
          s_acc[ni_local][0] = -FLT_MAX;
          s_acc[ni_local][2] = -FLT_MAX;
        }
        if (s_col1 >= kv_count) {
          s_acc[ni_local][1] = -FLT_MAX;
          s_acc[ni_local][3] = -FLT_MAX;
        }
      }
    }

    // Phases 1-2: Partial max within this warp half
    float partial_max0 = -FLT_MAX, partial_max1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_TILES_PER_WARP; ni++) {
      partial_max0 = fmaxf(partial_max0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      partial_max1 = fmaxf(partial_max1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int delta = 1; delta < 4; delta <<= 1) {
      partial_max0 =
          fmaxf(partial_max0, __shfl_xor_sync(0xFFFFFFFF, partial_max0, delta));
      partial_max1 =
          fmaxf(partial_max1, __shfl_xor_sync(0xFFFFFFFF, partial_max1, delta));
    }

    // Phase 3: Exchange partial max between warp halves via shared memory
    if (lane_id % 4 == 0) {
      smem_partial_max[warp_half * BLOCK_M + global_row0] = partial_max0;
      smem_partial_max[warp_half * BLOCK_M + global_row1] = partial_max1;
    }
    __syncthreads();

    float other_pmax0 =
        smem_partial_max[(1 - warp_half) * BLOCK_M + global_row0];
    float other_pmax1 =
        smem_partial_max[(1 - warp_half) * BLOCK_M + global_row1];
    float tile_max0 = fmaxf(partial_max0, other_pmax0);
    float tile_max1 = fmaxf(partial_max1, other_pmax1);
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, tile_max0);
    float new_max1 = fmaxf(prev_max1, tile_max1);
    float partial_sum0 = 0.0f, partial_sum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_TILES_PER_WARP; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][0] - new_max0)
                     : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][1] - new_max0)
                     : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][2] - new_max1)
                     : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][3] - new_max1)
                     : 0.0f;

      partial_sum0 += e0 + e1;
      partial_sum1 += e2 + e3;
      int ni_global = warp_half * QK_TILES_PER_WARP + ni;
      int p_col0 = ni_global * 8 + (lane_id % 4) * 2;
      int p_col1 = p_col0 + 1;
      smem_p[global_row0 * P_STRIDE + p_col0] = to_elem<T>(e0);
      smem_p[global_row0 * P_STRIDE + p_col1] = to_elem<T>(e1);
      smem_p[global_row1 * P_STRIDE + p_col0] = to_elem<T>(e2);
      smem_p[global_row1 * P_STRIDE + p_col1] = to_elem<T>(e3);
    }
#pragma unroll
    for (int delta = 1; delta < 4; delta <<= 1) {
      partial_sum0 += __shfl_xor_sync(0xFFFFFFFF, partial_sum0, delta);
      partial_sum1 += __shfl_xor_sync(0xFFFFFFFF, partial_sum1, delta);
    }
    cp_async_wait_group<0>();
    if (lane_id % 4 == 0) {
      smem_partial_sum[warp_half * BLOCK_M + global_row0] = partial_sum0;
      smem_partial_sum[warp_half * BLOCK_M + global_row1] = partial_sum1;
    }
    __syncthreads();

    float other_psum0 =
        smem_partial_sum[(1 - warp_half) * BLOCK_M + global_row0];
    float other_psum1 =
        smem_partial_sum[(1 - warp_half) * BLOCK_M + global_row1];
    float tile_sum0 = partial_sum0 + other_psum0;
    float tile_sum1 = partial_sum1 + other_psum1;
    {
      float corr0 = (kv_tile == 0) ? 0.0f : expf(prev_max0 - new_max0);
      float corr1 = (kv_tile == 0) ? 0.0f : expf(prev_max1 - new_max1);

      row_max0 = new_max0;
      row_max1 = new_max1;
      row_sum0 = row_sum0 * corr0 + tile_sum0;
      row_sum1 = row_sum1 * corr1 + tile_sum1;

#pragma unroll
      for (int di = 0; di < PV_TILES_PER_WARP; di++) {
        o_acc[di][0] *= corr0;
        o_acc[di][1] *= corr0;
        o_acc[di][2] *= corr1;
        o_acc[di][3] *= corr1;
      }
    }
    {
#pragma unroll
      for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
        int di = warp_half * PV_TILES_PER_WARP + di_local;

#pragma unroll
        for (int ki = 0; ki < TILES_BN; ki++) {
          // Load P tile (A operand)
          uint32_t a0, a1, a2, a3;
          {
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_p + (mi * 16 + row) * P_STRIDE + ki * 16 + col);
          }
          uint32_t b0, b1;
          {
            int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
            ldmatrix_x2_trans(b0, b1,
                              smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
          }

          ptx_mma_m16n8k16<T>(
              o_acc[di_local][0], o_acc[di_local][1], o_acc[di_local][2],
              o_acc[di_local][3], a0, a1, a2, a3, b0, b1, o_acc[di_local][0],
              o_acc[di_local][1], o_acc[di_local][2], o_acc[di_local][3]);
        }
      }
    }
    __syncthreads();
  } 
  {
    float inv_sum0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv_sum1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;

#pragma unroll
    for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
      int di = warp_half * PV_TILES_PER_WARP + di_local;
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;

      if (gq0 < seq_len) {
        O_head[gq0 * D_HEAD + col0] = to_elem<T>(o_acc[di_local][0] * inv_sum0);
        O_head[gq0 * D_HEAD + col1] = to_elem<T>(o_acc[di_local][1] * inv_sum0);
      }
      if (gq1 < seq_len) {
        O_head[gq1 * D_HEAD + col0] = to_elem<T>(o_acc[di_local][2] * inv_sum1);
        O_head[gq1 * D_HEAD + col1] = to_elem<T>(o_acc[di_local][3] * inv_sum1);
      }
    }

    // Optional: write log-sum-exp for backward pass or diagnostics
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_len)
        LSE[bh_idx * seq_len + gq0] = row_max0 + logf(fmaxf(row_sum0, 1e-10f));
      if (gq1 < seq_len)
        LSE[bh_idx * seq_len + gq1] = row_max1 + logf(fmaxf(row_sum1, 1e-10f));
    }
  }
}

template <class T> __device__ __forceinline__ uint32_t pack2(float a, float b);
template <> __device__ __forceinline__ uint32_t pack2<half>(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}
template <>
__device__ __forceinline__ uint32_t pack2<__nv_bfloat16>(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}

template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_fat_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE, const int seq_len,
    const int num_q_heads, const int num_kv_heads, const float scale) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "one m16 tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q stages through smem_k");

  const int bh_idx = blockIdx.y;
  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;
  if (q_start >= seq_len)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int QK_N8 = BLOCK_N / 8;     // score n8-tiles per warp (full BN)
  constexpr int TILES_K = D_HEAD / 16;   // k16-tiles for Q*K^T
  constexpr int PV_N8 = D_HEAD / 8;      // output n8-tiles per warp (full D)
  constexpr int TILES_BN = BLOCK_N / 16; // k16-tiles for P*V

  extern __shared__ char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);
  const size_t q_off = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
  const size_t kv_off =
      static_cast<size_t>(b * num_kv_heads + h_kv) * seq_len * D_HEAD;
  const T *Q_head = Q + q_off;
  const T *K_head = K + kv_off;
  const T *V_head = V + kv_off;
  T *O_head = O + q_off;

  const int mi = warp_id;
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;
  const float scale2 = scale * 1.4426950408889634f;
  uint32_t q_frag[TILES_K][4];
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val = (g < seq_len)
                      ? reinterpret_cast<const uint4 *>(Q_head + g * D_HEAD)[col]
                      : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++) {
      int row = lane_id % 16;
      int col = (lane_id / 16) * 8;
      ldmatrix_x4(q_frag[ki][0], q_frag[ki][1], q_frag[ki][2], q_frag[ki][3],
                  smem_k + (mi * 16 + row) * KV_STRIDE + ki * 16 + col);
    }
    __syncthreads(); // Q in regs; smem_k free for K
  }

  float o_acc[PV_N8][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  constexpr int VEC_COLS = D_HEAD / 8;
  auto issue_k = [&](int kv_start_i, int kv_count_i) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      bool valid = (row < kv_count_i);
      cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                  reinterpret_cast<const uint4 *>(
                      K_head + (size_t)(kv_start_i + row) * D_HEAD) +
                      col,
                  valid);
    }
    cp_async_commit_group();
  };
  issue_k(0, min(BLOCK_N, seq_len)); // prologue: K(0)

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_len - kv_start);

    cp_async_wait_group<0>(); // K(i) landed (only group in flight here)
    __syncthreads();          // K visible to all warps

    // V(i): streams behind QK + softmax.
    {
#pragma unroll
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        bool valid = (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(
                        V_head + (size_t)(kv_start + row) * D_HEAD) +
                        col,
                    valid);
      }
      cp_async_commit_group();
    }

    // -- Step A: S = Q * K^T, full BN per warp ------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        int k_row = lane_id % 8;
        int mat = (lane_id / 8) % 2;
        ldmatrix_x2(b0, b1,
                    smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 + mat * 8);
        ptx_mma_m16n8k16<T>(s_acc[ni][0], s_acc[ni][1], s_acc[ni][2],
                            s_acc[ni][3], q_frag[ki][0], q_frag[ki][1],
                            q_frag[ki][2], q_frag[ki][3], b0, b1, s_acc[ni][0],
                            s_acc[ni][1], s_acc[ni][2], s_acc[ni][3]);
      }
      int s_col0 = ni * 8 + (lane_id % 4) * 2;
      int s_col1 = s_col0 + 1;
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= scale2;
      if (CAUSAL) {
        if (kv_start + s_col0 > q_start + global_row0) s_acc[ni][0] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row0) s_acc[ni][1] = -FLT_MAX;
        if (kv_start + s_col0 > q_start + global_row1) s_acc[ni][2] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row1) s_acc[ni][3] = -FLT_MAX;
      }
      if (s_col0 >= kv_count) { s_acc[ni][0] = -FLT_MAX; s_acc[ni][2] = -FLT_MAX; }
      if (s_col1 >= kv_count) { s_acc[ni][1] = -FLT_MAX; s_acc[ni][3] = -FLT_MAX; }
    }

    // -- Step B: intra-warp softmax; exp overwrites s_acc (P stays in regs) --
    float pmax0 = -FLT_MAX, pmax1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      pmax0 = fmaxf(pmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      pmax1 = fmaxf(pmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFFu, pmax0, d));
      pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFFu, pmax1, d));
    }
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, pmax0);
    float new_max1 = fmaxf(prev_max1, pmax1);

    float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][0] - new_max0) : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][1] - new_max0) : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][2] - new_max1) : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][3] - new_max1) : 0.0f;
      psum0 += e0 + e1;
      psum1 += e2 + e3;
      s_acc[ni][0] = e0; s_acc[ni][1] = e1; s_acc[ni][2] = e2; s_acc[ni][3] = e3;
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      psum0 += __shfl_xor_sync(0xFFFFFFFFu, psum0, d);
      psum1 += __shfl_xor_sync(0xFFFFFFFFu, psum1, d);
    }

    // -- Step C: online correction -------------------------------------------
    float corr0 = (kv_tile == 0) ? 0.0f : exp2f(prev_max0 - new_max0);
    float corr1 = (kv_tile == 0) ? 0.0f : exp2f(prev_max1 - new_max1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + psum0;
    row_sum1 = row_sum1 * corr1 + psum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    cp_async_wait_group<0>(); // this thread's V arrived
    __syncthreads();          // all threads' V visible; K(i) dead CTA-wide

    // K(i+1): streams behind PV into the now-dead smem_k.
    if (kv_tile + 1 < num_kv_tiles) {
      int ks = (kv_tile + 1) * BLOCK_N;
      issue_k(ks, min(BLOCK_N, seq_len - ks));
    }
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      uint32_t a0 = pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]);
      uint32_t a1 = pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]);
      uint32_t a2 = pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]);
      uint32_t a3 = pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3]);
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        ldmatrix_x2_trans(b0, b1,
                          smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
        ptx_mma_m16n8k16<T>(o_acc[di][0], o_acc[di][1], o_acc[di][2],
                            o_acc[di][3], a0, a1, a2, a3, b0, b1, o_acc[di][0],
                            o_acc[di][1], o_acc[di][2], o_acc[di][3]);
      }
    }
    __syncthreads(); // smem_v dead before next tile's loads
  }

  // -- Finalize --------------------------------------------------------------
  {
    float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_len) {
        O_head[gq0 * D_HEAD + col0] = to_elem<T>(o_acc[di][0] * inv0);
        O_head[gq0 * D_HEAD + col1] = to_elem<T>(o_acc[di][1] * inv0);
      }
      if (gq1 < seq_len) {
        O_head[gq1 * D_HEAD + col0] = to_elem<T>(o_acc[di][2] * inv1);
        O_head[gq1 * D_HEAD + col1] = to_elem<T>(o_acc[di][3] * inv1);
      }
    }
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      // max/sum are in the base-2 domain (see scale2): LSE_e = ln2*(m2+log2 s2)
      constexpr float LN2 = 0.6931471805599453f;
      if (gq0 < seq_len)
        LSE[bh_idx * seq_len + gq0] =
            LN2 * (row_max0 + log2f(fmaxf(row_sum0, 1e-10f)));
      if (gq1 < seq_len)
        LSE[bh_idx * seq_len + gq1] =
            LN2 * (row_max1 + log2f(fmaxf(row_sum1, 1e-10f)));
    }
  }
}
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_fat_varlen_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE,
    const int *__restrict__ cu_seqlens_q, const int *__restrict__ cu_seqlens_k,
    const int num_q_heads, const int num_kv_heads, const float scale) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "one m16 tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q stages through smem_k");

  const int bh_idx = blockIdx.y;
  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);

  const int q0 = cu_seqlens_q[b];
  const int seq_q = cu_seqlens_q[b + 1] - q0;
  const int k0 = cu_seqlens_k[b];
  const int seq_k = cu_seqlens_k[b + 1] - k0;

  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;
  if (q_start >= seq_q)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int QK_N8 = BLOCK_N / 8;
  constexpr int TILES_K = D_HEAD / 16;
  constexpr int PV_N8 = D_HEAD / 8;
  constexpr int TILES_BN = BLOCK_N / 16;

  extern __shared__ char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  // Packed token-major addressing: token stride = H * D.
  const size_t q_row_stride = (size_t)num_q_heads * D_HEAD;
  const size_t kv_row_stride = (size_t)num_kv_heads * D_HEAD;
  const T *Q_head = Q + ((size_t)q0 * num_q_heads + h_q) * D_HEAD;
  const T *K_head = K + ((size_t)k0 * num_kv_heads + h_kv) * D_HEAD;
  const T *V_head = V + ((size_t)k0 * num_kv_heads + h_kv) * D_HEAD;
  T *O_head = O + ((size_t)q0 * num_q_heads + h_q) * D_HEAD;

  const int mi = warp_id;
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;

  const float scale2 = scale * 1.4426950408889634f;

  // Bottom-right causal alignment: query i attends kv j <= i + coff.
  const int coff = seq_k - seq_q;
  const int kv_end_raw = CAUSAL ? min(q_start + BLOCK_M + coff, seq_k) : seq_k;
  const int kv_end = max(kv_end_raw, 0);
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  // -- Q prologue: stage through smem_k, ldmatrix into q_frag ----------------
  uint32_t q_frag[TILES_K][4];
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val =
          (g < seq_q)
              ? reinterpret_cast<const uint4 *>(Q_head + g * q_row_stride)[col]
              : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++) {
      int row = lane_id % 16;
      int col = (lane_id / 16) * 8;
      ldmatrix_x4(q_frag[ki][0], q_frag[ki][1], q_frag[ki][2], q_frag[ki][3],
                  smem_k + (mi * 16 + row) * KV_STRIDE + ki * 16 + col);
    }
    __syncthreads(); // Q in regs; smem_k free for K
  }

  float o_acc[PV_N8][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  constexpr int VEC_COLS = D_HEAD / 8;
  auto issue_k = [&](int kv_start_i, int kv_count_i) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      bool valid = (row < kv_count_i);
      cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                  reinterpret_cast<const uint4 *>(
                      K_head + (size_t)(kv_start_i + row) * kv_row_stride) +
                      col,
                  valid);
    }
    cp_async_commit_group();
  };
  if (num_kv_tiles > 0)
    issue_k(0, min(BLOCK_N, seq_k));

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_k - kv_start);

    cp_async_wait_group<0>(); // K(i) landed
    __syncthreads();          // K visible to all warps

    // V(i): streams behind QK + softmax.
    {
#pragma unroll
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        bool valid = (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(
                        V_head + (size_t)(kv_start + row) * kv_row_stride) +
                        col,
                    valid);
      }
      cp_async_commit_group();
    }

    // -- Step A: S = Q * K^T -------------------------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        int k_row = lane_id % 8;
        int mat = (lane_id / 8) % 2;
        ldmatrix_x2(b0, b1,
                    smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 + mat * 8);
        ptx_mma_m16n8k16<T>(s_acc[ni][0], s_acc[ni][1], s_acc[ni][2],
                            s_acc[ni][3], q_frag[ki][0], q_frag[ki][1],
                            q_frag[ki][2], q_frag[ki][3], b0, b1, s_acc[ni][0],
                            s_acc[ni][1], s_acc[ni][2], s_acc[ni][3]);
      }
      int s_col0 = ni * 8 + (lane_id % 4) * 2;
      int s_col1 = s_col0 + 1;
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= scale2;
      if (CAUSAL) {
        if (kv_start + s_col0 > q_start + global_row0 + coff) s_acc[ni][0] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row0 + coff) s_acc[ni][1] = -FLT_MAX;
        if (kv_start + s_col0 > q_start + global_row1 + coff) s_acc[ni][2] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row1 + coff) s_acc[ni][3] = -FLT_MAX;
      }
      if (s_col0 >= kv_count) { s_acc[ni][0] = -FLT_MAX; s_acc[ni][2] = -FLT_MAX; }
      if (s_col1 >= kv_count) { s_acc[ni][1] = -FLT_MAX; s_acc[ni][3] = -FLT_MAX; }
    }

    // -- Step B: intra-warp softmax (exp2 domain; P stays in registers) ------
    float pmax0 = -FLT_MAX, pmax1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      pmax0 = fmaxf(pmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      pmax1 = fmaxf(pmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFFu, pmax0, d));
      pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFFu, pmax1, d));
    }
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, pmax0);
    float new_max1 = fmaxf(prev_max1, pmax1);

    float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][0] - new_max0) : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][1] - new_max0) : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][2] - new_max1) : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][3] - new_max1) : 0.0f;
      psum0 += e0 + e1;
      psum1 += e2 + e3;
      s_acc[ni][0] = e0; s_acc[ni][1] = e1; s_acc[ni][2] = e2; s_acc[ni][3] = e3;
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      psum0 += __shfl_xor_sync(0xFFFFFFFFu, psum0, d);
      psum1 += __shfl_xor_sync(0xFFFFFFFFu, psum1, d);
    }

    // -- Step C: online correction -------------------------------------------
    float corr0 = (kv_tile == 0) ? 0.0f : exp2f(prev_max0 - new_max0);
    float corr1 = (kv_tile == 0) ? 0.0f : exp2f(prev_max1 - new_max1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + psum0;
    row_sum1 = row_sum1 * corr1 + psum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    cp_async_wait_group<0>(); // V arrived
    __syncthreads();          // V visible; K(i) dead CTA-wide

    if (kv_tile + 1 < num_kv_tiles) {
      int ks = (kv_tile + 1) * BLOCK_N;
      issue_k(ks, min(BLOCK_N, seq_k - ks));
    }

    // -- Step D: O += P * V (A operand packed from s_acc) --------------------
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      uint32_t a0 = pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]);
      uint32_t a1 = pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]);
      uint32_t a2 = pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]);
      uint32_t a3 = pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3]);
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        ldmatrix_x2_trans(b0, b1,
                          smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
        ptx_mma_m16n8k16<T>(o_acc[di][0], o_acc[di][1], o_acc[di][2],
                            o_acc[di][3], a0, a1, a2, a3, b0, b1, o_acc[di][0],
                            o_acc[di][1], o_acc[di][2], o_acc[di][3]);
      }
    }
    __syncthreads(); // smem_v dead before next tile's loads
  }

  // -- Finalize ---------------------------------------------------------------
  {
    float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_q) {
        O_head[gq0 * q_row_stride + col0] = to_elem<T>(o_acc[di][0] * inv0);
        O_head[gq0 * q_row_stride + col1] = to_elem<T>(o_acc[di][1] * inv0);
      }
      if (gq1 < seq_q) {
        O_head[gq1 * q_row_stride + col0] = to_elem<T>(o_acc[di][2] * inv1);
        O_head[gq1 * q_row_stride + col1] = to_elem<T>(o_acc[di][3] * inv1);
      }
    }
    if (lane_id % 4 == 0 && LSE != nullptr) {
      // max/sum live in the base-2 domain (scale2): LSE_e = ln2*(m2 + log2 s2)
      constexpr float LN2 = 0.6931471805599453f;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_q)
        LSE[(size_t)(q0 + gq0) * num_q_heads + h_q] =
            (row_sum0 > 0.0f)
                ? LN2 * (row_max0 + log2f(row_sum0))
                : -FLT_MAX;
      if (gq1 < seq_q)
        LSE[(size_t)(q0 + gq1) * num_q_heads + h_q] =
            (row_sum1 > 0.0f)
                ? LN2 * (row_max1 + log2f(row_sum1))
                : -FLT_MAX;
    }
  }
}
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_fat_paged_prefill_kernel(
    const T *__restrict__ Q, const T *__restrict__ Kpool,
    const T *__restrict__ Vpool, T *__restrict__ O, float *__restrict__ LSE,
    const int *__restrict__ cu_seqlens_q, const int *__restrict__ seq_lens_k,
    const int *__restrict__ block_table, const int max_blocks,
    const int page_size, const int num_q_heads, const int num_kv_heads,
    const float scale) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "one m16 tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q stages through smem_k");

  const int bh_idx = blockIdx.y;
  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);

  const int q0 = cu_seqlens_q[b];
  const int seq_q = cu_seqlens_q[b + 1] - q0;
  const int seq_k = seq_lens_k[b];
  const int *bt = block_table + (size_t)b * max_blocks;

  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;
  if (q_start >= seq_q)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int QK_N8 = BLOCK_N / 8;
  constexpr int TILES_K = D_HEAD / 16;
  constexpr int PV_N8 = D_HEAD / 8;
  constexpr int TILES_BN = BLOCK_N / 16;

  extern __shared__ char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  const size_t q_row_stride = (size_t)num_q_heads * D_HEAD;
  const T *Q_head = Q + ((size_t)q0 * num_q_heads + h_q) * D_HEAD;
  T *O_head = O + ((size_t)q0 * num_q_heads + h_q) * D_HEAD;
  // pool row base for this kv head; token row = ((page*ps + slot)*H_kv)*D
  const T *K_head = Kpool + (size_t)h_kv * D_HEAD;
  const T *V_head = Vpool + (size_t)h_kv * D_HEAD;
  const size_t pool_row_stride = (size_t)num_kv_heads * D_HEAD;

  const int mi = warp_id;
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;

  const float scale2 = scale * 1.4426950408889634f;

  const int coff = seq_k - seq_q;
  const int kv_end_raw = CAUSAL ? min(q_start + BLOCK_M + coff, seq_k) : seq_k;
  const int kv_end = max(kv_end_raw, 0);
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  // -- Q prologue -------------------------------------------------------------
  uint32_t q_frag[TILES_K][4];
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val =
          (g < seq_q)
              ? reinterpret_cast<const uint4 *>(Q_head + g * q_row_stride)[col]
              : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++) {
      int row = lane_id % 16;
      int col = (lane_id / 16) * 8;
      ldmatrix_x4(q_frag[ki][0], q_frag[ki][1], q_frag[ki][2], q_frag[ki][3],
                  smem_k + (mi * 16 + row) * KV_STRIDE + ki * 16 + col);
    }
    __syncthreads();
  }

  float o_acc[PV_N8][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  constexpr int VEC_COLS = D_HEAD / 8;
  // per-row page translation for the cp.async source
  auto kv_src = [&](const T *pool_head, int token) -> const T * {
    int blk = token / page_size;
    int slot = token - blk * page_size;
    return pool_head + ((size_t)bt[blk] * page_size + slot) * pool_row_stride;
  };
  // Invalid rows still need an in-bounds ADDRESS for the predicated cp.async
  // (src_size=0 zero-fills, but the address goes through the block table) —
  // clamp to the tile's last valid token. kv_count_i >= 1 whenever a tile is
  // issued.
  auto issue_k = [&](int kv_start_i, int kv_count_i) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      bool valid = (row < kv_count_i);
      cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                  reinterpret_cast<const uint4 *>(
                      kv_src(K_head, kv_start_i + min(row, kv_count_i - 1))) +
                      col,
                  valid);
    }
    cp_async_commit_group();
  };
  if (num_kv_tiles > 0)
    issue_k(0, min(BLOCK_N, seq_k));

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_k - kv_start);

    cp_async_wait_group<0>();
    __syncthreads();

    // V(i)
    {
#pragma unroll
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        bool valid = (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(kv_src(
                        V_head, kv_start + (valid ? row : kv_count - 1))) +
                        col,
                    valid);
      }
      cp_async_commit_group();
    }

    // -- Step A ----------------------------------------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        int k_row = lane_id % 8;
        int mat = (lane_id / 8) % 2;
        ldmatrix_x2(b0, b1,
                    smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 + mat * 8);
        ptx_mma_m16n8k16<T>(s_acc[ni][0], s_acc[ni][1], s_acc[ni][2],
                            s_acc[ni][3], q_frag[ki][0], q_frag[ki][1],
                            q_frag[ki][2], q_frag[ki][3], b0, b1, s_acc[ni][0],
                            s_acc[ni][1], s_acc[ni][2], s_acc[ni][3]);
      }
      int s_col0 = ni * 8 + (lane_id % 4) * 2;
      int s_col1 = s_col0 + 1;
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= scale2;
      if (CAUSAL) {
        if (kv_start + s_col0 > q_start + global_row0 + coff) s_acc[ni][0] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row0 + coff) s_acc[ni][1] = -FLT_MAX;
        if (kv_start + s_col0 > q_start + global_row1 + coff) s_acc[ni][2] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row1 + coff) s_acc[ni][3] = -FLT_MAX;
      }
      if (s_col0 >= kv_count) { s_acc[ni][0] = -FLT_MAX; s_acc[ni][2] = -FLT_MAX; }
      if (s_col1 >= kv_count) { s_acc[ni][1] = -FLT_MAX; s_acc[ni][3] = -FLT_MAX; }
    }

    // -- Step B ----------------------------------------------------------------
    float pmax0 = -FLT_MAX, pmax1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      pmax0 = fmaxf(pmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      pmax1 = fmaxf(pmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFFu, pmax0, d));
      pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFFu, pmax1, d));
    }
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, pmax0);
    float new_max1 = fmaxf(prev_max1, pmax1);

    float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][0] - new_max0) : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][1] - new_max0) : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][2] - new_max1) : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][3] - new_max1) : 0.0f;
      psum0 += e0 + e1;
      psum1 += e2 + e3;
      s_acc[ni][0] = e0; s_acc[ni][1] = e1; s_acc[ni][2] = e2; s_acc[ni][3] = e3;
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      psum0 += __shfl_xor_sync(0xFFFFFFFFu, psum0, d);
      psum1 += __shfl_xor_sync(0xFFFFFFFFu, psum1, d);
    }

    // -- Step C ----------------------------------------------------------------
    float corr0 = (kv_tile == 0) ? 0.0f : exp2f(prev_max0 - new_max0);
    float corr1 = (kv_tile == 0) ? 0.0f : exp2f(prev_max1 - new_max1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + psum0;
    row_sum1 = row_sum1 * corr1 + psum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    cp_async_wait_group<0>();
    __syncthreads();

    if (kv_tile + 1 < num_kv_tiles) {
      int ks = (kv_tile + 1) * BLOCK_N;
      issue_k(ks, min(BLOCK_N, seq_k - ks));
    }

    // -- Step D ----------------------------------------------------------------
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      uint32_t a0 = pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]);
      uint32_t a1 = pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]);
      uint32_t a2 = pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]);
      uint32_t a3 = pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3]);
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        ldmatrix_x2_trans(b0, b1,
                          smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
        ptx_mma_m16n8k16<T>(o_acc[di][0], o_acc[di][1], o_acc[di][2],
                            o_acc[di][3], a0, a1, a2, a3, b0, b1, o_acc[di][0],
                            o_acc[di][1], o_acc[di][2], o_acc[di][3]);
      }
    }
    __syncthreads();
  }

  // -- Finalize ----------------------------------------------------------------
  {
    float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_q) {
        O_head[gq0 * q_row_stride + col0] = to_elem<T>(o_acc[di][0] * inv0);
        O_head[gq0 * q_row_stride + col1] = to_elem<T>(o_acc[di][1] * inv0);
      }
      if (gq1 < seq_q) {
        O_head[gq1 * q_row_stride + col0] = to_elem<T>(o_acc[di][2] * inv1);
        O_head[gq1 * q_row_stride + col1] = to_elem<T>(o_acc[di][3] * inv1);
      }
    }
    if (lane_id % 4 == 0 && LSE != nullptr) {
      constexpr float LN2 = 0.6931471805599453f;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_q)
        LSE[(size_t)(q0 + gq0) * num_q_heads + h_q] =
            (row_sum0 > 0.0f) ? LN2 * (row_max0 + log2f(row_sum0)) : -FLT_MAX;
      if (gq1 < seq_q)
        LSE[(size_t)(q0 + gq1) * num_q_heads + h_q] =
            (row_sum1 > 0.0f) ? LN2 * (row_max1 + log2f(row_sum1)) : -FLT_MAX;
    }
  }
}
namespace {

// Common launch path templated on tile geometry. Computes smem, opts in if
// needed, dispatches on the causal flag.
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, class T>
inline void launch_variant(const FlashAttentionParams &params) {
  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;

  const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
  const int grid_y = params.batch_size * params.num_heads;
  dim3 grid(grid_x, grid_y);
  dim3 block(WARP_SIZE_FA, NUM_WARPS);

  size_t smem_bytes = 0;
  smem_bytes += BLOCK_M * Q_STRIDE * sizeof(T); // smem_q
  smem_bytes +=
      BLOCK_N * KV_STRIDE * sizeof(T); // smem_k (also holds P, aliased)
  smem_bytes += BLOCK_N * KV_STRIDE * sizeof(T); // smem_v
  // smem_p is aliased onto smem_k (see kernel) — no separate allocation.
  smem_bytes += 4 * BLOCK_M * sizeof(float); // partial_max + partial_sum

  if (smem_bytes > 48 * 1024) {
    if (params.causal) {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true,
                                     T>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    } else {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false,
                                     T>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
  }

  // GQA: num_kv_heads==0 means MHA (KV head count == query head count).
  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  } else {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}
template <int D_HEAD, class T>
inline void launch_fat_variant(const FlashAttentionParams &params) {
  constexpr int BM = 64, BN = 64, NW = 4;
  constexpr int KV_STRIDE = D_HEAD + 8;

  const int grid_x = (params.seq_len + BM - 1) / BM;
  dim3 grid(grid_x, params.batch_size * params.num_heads);
  dim3 block(WARP_SIZE_FA, NW);
  size_t smem_bytes = 2 * (size_t)BN * KV_STRIDE * sizeof(T);

  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  } else {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}
template <int D_HEAD, class T>
inline void launch_fat_varlen(const FlashAttentionVarlenParams &params) {
  constexpr int BM = 64, BN = 64, NW = 4;
  constexpr int KV_STRIDE = D_HEAD + 8;

  const int grid_x = (params.max_seqlen_q + BM - 1) / BM;
  dim3 grid(grid_x, params.batch_size * params.num_heads);
  dim3 block(WARP_SIZE_FA, NW);
  size_t smem_bytes = 2 * (size_t)BN * KV_STRIDE * sizeof(T);

  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_fat_varlen_kernel<BM, BN, D_HEAD, NW, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.cu_seqlens_q, params.cu_seqlens_k,
            H_q, H_kv, params.scale);
  } else {
    flash_attention_fat_varlen_kernel<BM, BN, D_HEAD, NW, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.cu_seqlens_q, params.cu_seqlens_k,
            H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <int D_HEAD, class T>
inline void
launch_fat_paged_prefill(const FlashAttentionPagedPrefillParams &params) {
  constexpr int BM = 64, BN = 64, NW = 4;
  constexpr int KV_STRIDE = D_HEAD + 8;

  const int grid_x = (params.max_seqlen_q + BM - 1) / BM;
  dim3 grid(grid_x, params.batch_size * params.num_heads);
  dim3 block(WARP_SIZE_FA, NW);
  size_t smem_bytes = 2 * (size_t)BN * KV_STRIDE * sizeof(T);

  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K_cache);
  const T *Vp = reinterpret_cast<const T *>(params.V_cache);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_fat_paged_prefill_kernel<BM, BN, D_HEAD, NW, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.cu_seqlens_q, params.seq_lens_k,
            params.block_table, params.max_blocks_per_seq, params.page_size,
            H_q, H_kv, params.scale);
  } else {
    flash_attention_fat_paged_prefill_kernel<BM, BN, D_HEAD, NW, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.cu_seqlens_q, params.seq_lens_k,
            params.block_table, params.max_blocks_per_seq, params.page_size,
            H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

inline int get_sm_count() {
  static int sm_count = -1;
  if (sm_count < 0) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  }
  return sm_count;
}
template <int D> struct FaConfig;
template <> struct FaConfig<64> {
  static constexpr int BN = 64, BM_SMALL = 32, W_SMALL = 4, SAT_MULT = 3;
};
template <> struct FaConfig<128> {
  static constexpr int BN = 32, BM_SMALL = 32, W_SMALL = 4, SAT_MULT = 2;
};
template <class T, int D_HEAD>
inline void dispatch_by_saturation(const FlashAttentionParams &params) {
  using C = FaConfig<D_HEAD>;
  constexpr int FAT_BM = 64; // fat-warp kernel row-tile height
  const int num_blocks_fat = params.batch_size * params.num_heads *
                             ((params.seq_len + FAT_BM - 1) / FAT_BM);
  const int sm_count = get_sm_count();

  if (num_blocks_fat < C::SAT_MULT * sm_count) {
    launch_variant<C::BM_SMALL, C::BN, D_HEAD, C::W_SMALL, T>(params);
  } else {
    launch_fat_variant<D_HEAD, T>(params);
  }
}
template <int BM, int BN, int D, class T> constexpr size_t fa_cfg_smem() {
  constexpr int PAD = 8; // matches launch_variant's SMEM_PAD
  return (size_t)BM * (D + PAD) * sizeof(T)       // smem_q
         + (size_t)BN * (D + PAD) * sizeof(T)     // smem_k (P aliased)
         + (size_t)BN * (D + PAD) * sizeof(T)     // smem_v
         + 4 * (size_t)BM * sizeof(float);        // partials
}
// Fat kernel smem: K + V tiles only (Q lives in registers, P never lands).
template <int D, class T> constexpr size_t fa_fat_smem() {
  return 2 * (size_t)64 * (D + 8) * sizeof(T);
}
template <class T> const FaCandidate *fa_configs_64(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_fat_variant<64, T>, fa_fat_smem<64, T>()},
      {32, 64, 4, &launch_variant<32, 64, 64, 4, T>, fa_cfg_smem<32, 64, 64, T>()},
      {32, 32, 4, &launch_variant<32, 32, 64, 4, T>, fa_cfg_smem<32, 32, 64, T>()},
  };
  n = 3;
  return c;
}
template <class T> const FaCandidate *fa_configs_128(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_fat_variant<128, T>, fa_fat_smem<128, T>()},
      {32, 32, 4, &launch_variant<32, 32, 128, 4, T>, fa_cfg_smem<32, 32, 128, T>()},
      {32, 64, 4, &launch_variant<32, 64, 128, 4, T>, fa_cfg_smem<32, 64, 128, T>()},
  };
  n = 3;
  return c;
}

template <class T, int D>
inline void launch_autotuned(const FlashAttentionParams &p) {
  int n = 0;
  const FaCandidate *cands;
  if constexpr (D == 64)
    cands = fa_configs_64<T>(n);
  else
    cands = fa_configs_128<T>(n);
  const int idx = fa_autotune_pick(cands, n, p);
  cands[idx].launch(p);
}

} // anonymous namespace

// Runtime (dtype, head-dim) -> compile-time instantiation. d_head must be a
// multiple of 16 (the MMA k-tile); 64 and 128 are the tuned paths (GLM, Llama,
// DeepSeek-LLM, most decoders). dtype FP16 or BF16 (BF16 for Llama-3/Mistral/
// Qwen/GLM, which ship bf16 weights). 80/96 are reachable by adding cases.
void launch_flash_attention(const FlashAttentionParams &params) {
  const bool bf16 = (params.dtype == DType::BF16);
  // Opt-in autotuner: benchmark candidate tile configs once per (shape,dtype),
  // cache the fastest. Default (autotune==false) uses the heuristic below.
  if (params.autotune && (params.d_head == 64 || params.d_head == 128)) {
    if (params.d_head == 64) {
      if (bf16)
        launch_autotuned<__nv_bfloat16, 64>(params);
      else
        launch_autotuned<half, 64>(params);
    } else {
      if (bf16)
        launch_autotuned<__nv_bfloat16, 128>(params);
      else
        launch_autotuned<half, 128>(params);
    }
    return;
  }
  switch (params.d_head) {
  case 64:
    if (bf16)
      dispatch_by_saturation<__nv_bfloat16, 64>(params);
    else
      dispatch_by_saturation<half, 64>(params);
    break;
  case 128:
    if (bf16)
      dispatch_by_saturation<__nv_bfloat16, 128>(params);
    else
      dispatch_by_saturation<half, 128>(params);
    break;
  default:
    fprintf(stderr, "flash_attention: unsupported d_head=%d (tuned: 64, 128)\n",
            params.d_head);
    abort();
  }
}
void launch_flash_attention_varlen(const FlashAttentionVarlenParams &params) {
  const bool bf16 = (params.dtype == DType::BF16);
  switch (params.d_head) {
  case 64:
    if (bf16)
      launch_fat_varlen<64, __nv_bfloat16>(params);
    else
      launch_fat_varlen<64, half>(params);
    break;
  case 128:
    if (bf16)
      launch_fat_varlen<128, __nv_bfloat16>(params);
    else
      launch_fat_varlen<128, half>(params);
    break;
  default:
    fprintf(stderr,
            "flash_attention_varlen: unsupported d_head=%d (tuned: 64, 128)\n",
            params.d_head);
    abort();
  }
}
void launch_flash_attention_paged_prefill(
    const FlashAttentionPagedPrefillParams &params) {
  if (params.page_size <= 0 || params.max_blocks_per_seq <= 0) {
    fprintf(stderr,
            "flash_attention_paged_prefill: bad page geometry (page_size=%d, "
            "max_blocks_per_seq=%d)\n",
            params.page_size, params.max_blocks_per_seq);
    abort();
  }
  const bool bf16 = (params.dtype == DType::BF16);
  switch (params.d_head) {
  case 64:
    if (bf16)
      launch_fat_paged_prefill<64, __nv_bfloat16>(params);
    else
      launch_fat_paged_prefill<64, half>(params);
    break;
  case 128:
    if (bf16)
      launch_fat_paged_prefill<128, __nv_bfloat16>(params);
    else
      launch_fat_paged_prefill<128, half>(params);
    break;
  default:
    fprintf(stderr,
            "flash_attention_paged_prefill: unsupported d_head=%d (tuned: 64, "
            "128)\n",
            params.d_head);
    abort();
  }
}

} // namespace transformer