# SPDX-License-Identifier: Apache-2.0
# Based on Minghua Liu's MSN auction-EMD implementation, via PD-LTS.
# Modified for Jittor inline CUDA and intermediate memory management.
"""失败对照：auction-EMD kernel 移植（Jittor `jt.code` 内联 CUDA）。

逐条对齐 上游 PyTorch 参考实现/metric/emd/emd_cuda.cu 的 6 个
kernel（clear/calc_unass_cnt/calc_unass_cnt_sum/calc_unass_idx/Bid/GetMax/
Assign/CalcDist）+ backward 的 NmDistanceGradKernel，算法逐字不改，只把
Python 侧预分配的中间状态数组改为 kernel 内部 `cudaMalloc` 自管理。

约束（沿用原实现，未放宽）：
    - batch_size <= 512（`calc_unass_cnt_sum` 用共享内存数组硬编码 512）
    - n % 1024 == 0
    - 输入需归一化到合理数值范围（原实现假定坐标在 [0,1]，`Bid` kernel 里
      `3.0 - dist - price` 的 3.0 是坐标差平方和的理论上界系数）
    - 只对 xyz1（预测点云）有梯度，xyz2（GT）无梯度（原实现的已知限制）

用法：
    from emd_jittor import earth_mover_distance
    dist, assignment = earth_mover_distance(xyz1, xyz2, eps=0.005, iters=50)
    loss = dist.sum()
"""

from __future__ import annotations

import jittor as jt
from jittor import Function

_CUDA_HEADER = """
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float emd_atomicMax(float *address, float val) {
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__ void emd_clear(int b, int *cnt_tmp, int *unass_cnt) {
    for (int i = threadIdx.x; i < b; i += blockDim.x) {
        cnt_tmp[i] = 0;
        unass_cnt[i] = 0;
    }
}

__global__ void emd_calc_unass_cnt(int b, int n, int *assignment, int *unass_cnt) {
    const int BLOCK_SIZE = 1024;
    __shared__ int scan_array[BLOCK_SIZE];
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        scan_array[threadIdx.x] = assignment[i * n + blockIdx.y * BLOCK_SIZE + threadIdx.x] == -1 ? 1 : 0;
        __syncthreads();
        int stride = 1;
        while (stride <= BLOCK_SIZE / 2) {
            int index = (threadIdx.x + 1) * stride * 2 - 1;
            if (index < BLOCK_SIZE)
                scan_array[index] += scan_array[index - stride];
            stride = stride * 2;
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == BLOCK_SIZE - 1) {
            atomicAdd(&unass_cnt[i], scan_array[threadIdx.x]);
        }
        __syncthreads();
    }
}

__global__ void emd_calc_unass_cnt_sum(int b, int *unass_cnt, int *unass_cnt_sum) {
    const int BLOCK_SIZE = 512;
    __shared__ int scan_array[BLOCK_SIZE];
    scan_array[threadIdx.x] = unass_cnt[threadIdx.x];
    __syncthreads();
    int stride = 1;
    while (stride <= BLOCK_SIZE / 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE)
            scan_array[index] += scan_array[index - stride];
        stride = stride * 2;
        __syncthreads();
    }
    __syncthreads();
    stride = BLOCK_SIZE / 4;
    while (stride > 0) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if ((index + stride) < BLOCK_SIZE)
            scan_array[index + stride] += scan_array[index];
        stride = stride / 2;
        __syncthreads();
    }
    __syncthreads();
    unass_cnt_sum[threadIdx.x] = scan_array[threadIdx.x];
}

__global__ void emd_calc_unass_idx(int b, int n, int *assignment, int *unass_idx, int *unass_cnt,
                                    int *unass_cnt_sum, int *cnt_tmp) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        if (assignment[i * n + blockIdx.y * 1024 + threadIdx.x] == -1) {
            int idx = atomicAdd(&cnt_tmp[i], 1);
            unass_idx[unass_cnt_sum[i] - unass_cnt[i] + idx] = blockIdx.y * 1024 + threadIdx.x;
        }
    }
}

__global__ void emd_Bid(int b, int n, const float *xyz1, const float *xyz2, float eps, int *assignment,
                         int *assignment_inv, float *price, int *bid, float *bid_increments,
                         float *max_increments, int *unass_cnt, int *unass_cnt_sum, int *unass_idx) {
    const int batch = 2048, block_size = 1024, block_cnt = n / 1024;
    __shared__ float xyz2_buf[batch * 3];
    __shared__ float price_buf[batch];
    __shared__ float best_buf[block_size];
    __shared__ float better_buf[block_size];
    __shared__ int best_i_buf[block_size];
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int _unass_cnt = unass_cnt[i];
        if (_unass_cnt == 0)
            continue;
        int _unass_cnt_sum = unass_cnt_sum[i];
        int unass_per_block = (_unass_cnt + block_cnt - 1) / block_cnt;
        int thread_per_unass = block_size / unass_per_block;
        int unass_this_block = max(min(_unass_cnt - (int)blockIdx.y * unass_per_block, unass_per_block), 0);

        float x1, y1, z1, best = -1e9, better = -1e9;
        int best_i = -1, _unass_id = -1, thread_in_unass = 0;

        if (threadIdx.x < thread_per_unass * unass_this_block) {
            _unass_id = unass_per_block * blockIdx.y + threadIdx.x / thread_per_unass + _unass_cnt_sum - _unass_cnt;
            _unass_id = unass_idx[_unass_id];
            thread_in_unass = threadIdx.x % thread_per_unass;

            x1 = xyz1[(i * n + _unass_id) * 3 + 0];
            y1 = xyz1[(i * n + _unass_id) * 3 + 1];
            z1 = xyz1[(i * n + _unass_id) * 3 + 2];
        }

        for (int k2 = 0; k2 < n; k2 += batch) {
            int end_k = min(n, k2 + batch) - k2;
            for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
                xyz2_buf[j] = xyz2[(i * n + k2) * 3 + j];
            }
            for (int j = threadIdx.x; j < end_k; j += blockDim.x) {
                price_buf[j] = price[i * n + k2 + j];
            }
            __syncthreads();

            if (_unass_id != -1) {
                int delta = (end_k + thread_per_unass - 1) / thread_per_unass;
                int l = thread_in_unass * delta;
                int r = min((thread_in_unass + 1) * delta, end_k);
                for (int k = l; k < r; k++) {
                    float x2 = xyz2_buf[k * 3 + 0] - x1;
                    float y2 = xyz2_buf[k * 3 + 1] - y1;
                    float z2 = xyz2_buf[k * 3 + 2] - z1;
                    float d = 3.0 - sqrtf(x2 * x2 + y2 * y2 + z2 * z2) - price_buf[k];
                    if (d > best) {
                        better = best;
                        best = d;
                        best_i = k + k2;
                    } else if (d > better) {
                        better = d;
                    }
                }
            }
            __syncthreads();
        }

        best_buf[threadIdx.x] = best;
        better_buf[threadIdx.x] = better;
        best_i_buf[threadIdx.x] = best_i;
        __syncthreads();

        if (_unass_id != -1 && thread_in_unass == 0) {
            for (int j = threadIdx.x + 1; j < threadIdx.x + thread_per_unass; j++) {
                if (best_buf[j] > best) {
                    better = max(best, better_buf[j]);
                    best = best_buf[j];
                    best_i = best_i_buf[j];
                } else
                    better = max(better, best_buf[j]);
            }
            bid[i * n + _unass_id] = best_i;
            bid_increments[i * n + _unass_id] = best - better + eps;
            emd_atomicMax(&max_increments[i * n + best_i], best - better + eps);
        }
    }
}

__global__ void emd_GetMax(int b, int n, int *assignment, int *bid, float *bid_increments,
                            float *max_increments, int *max_idx) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int j = threadIdx.x + blockIdx.y * blockDim.x;
        if (assignment[i * n + j] == -1) {
            int bid_id = bid[i * n + j];
            float bid_inc = bid_increments[i * n + j];
            float max_inc = max_increments[i * n + bid_id];
            if (bid_inc - 1e-6 <= max_inc && max_inc <= bid_inc + 1e-6) {
                max_idx[i * n + bid_id] = j;
            }
        }
    }
}

__global__ void emd_Assign(int b, int n, int *assignment, int *assignment_inv, float *price, int *bid,
                            float *bid_increments, float *max_increments, int *max_idx, bool last) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int j = threadIdx.x + blockIdx.y * blockDim.x;
        if (assignment[i * n + j] == -1) {
            int bid_id = bid[i * n + j];
            if (last || max_idx[i * n + bid_id] == j) {
                float bid_inc = bid_increments[i * n + j];
                int ass_inv = assignment_inv[i * n + bid_id];
                if (!last && ass_inv != -1) {
                    assignment[i * n + ass_inv] = -1;
                }
                assignment_inv[i * n + bid_id] = j;
                assignment[i * n + j] = bid_id;
                price[i * n + bid_id] += bid_inc;
                max_increments[i * n + bid_id] = -1e9;
            }
        }
    }
}

__global__ void emd_CalcDist(int b, int n, float *xyz1, float *xyz2, float *dist, int *assignment) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int j = threadIdx.x + blockIdx.y * blockDim.x;
        int k = assignment[i * n + j];
        float deltax = xyz1[(i * n + j) * 3 + 0] - xyz2[(i * n + k) * 3 + 0];
        float deltay = xyz1[(i * n + j) * 3 + 1] - xyz2[(i * n + k) * 3 + 1];
        float deltaz = xyz1[(i * n + j) * 3 + 2] - xyz2[(i * n + k) * 3 + 2];
        dist[i * n + j] = deltax * deltax + deltay * deltay + deltaz * deltaz;
    }
}

__global__ void emd_grad_kernel(int b, int n, const float *xyz1, const float *xyz2, const float *grad_dist,
                                 const int *idx, float *grad_xyz) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
            float x1 = xyz1[(i * n + j) * 3 + 0];
            float y1 = xyz1[(i * n + j) * 3 + 1];
            float z1 = xyz1[(i * n + j) * 3 + 2];
            int j2 = idx[i * n + j];
            float x2 = xyz2[(i * n + j2) * 3 + 0];
            float y2 = xyz2[(i * n + j2) * 3 + 1];
            float z2 = xyz2[(i * n + j2) * 3 + 2];
            float g = grad_dist[i * n + j] * 2;
            atomicAdd(&(grad_xyz[(i * n + j) * 3 + 0]), g * (x1 - x2));
            atomicAdd(&(grad_xyz[(i * n + j) * 3 + 1]), g * (y1 - y2));
            atomicAdd(&(grad_xyz[(i * n + j) * 3 + 2]), g * (z1 - z2));
        }
    }
}
"""


def _forward_cuda_src(eps: float, iters: int) -> str:
    return f"""
int batch_size = in0->shape[0];
int n = in0->shape[1];
ASSERT(n % 1024 == 0) << "n must be a multiple of 1024, got " << n;
ASSERT(batch_size <= 512) << "batch_size must be <= 512, got " << batch_size;

// out1_p (assignment, int32) 直接作为算法状态数组使用，不额外分配/拷贝。
int *assignment_inv, *bid, *unass_idx, *unass_cnt, *unass_cnt_sum, *cnt_tmp, *max_idx;
float *price, *bid_increments, *max_increments;

cudaMalloc(&assignment_inv, batch_size * n * sizeof(int));
cudaMalloc(&bid, batch_size * n * sizeof(int));
cudaMalloc(&unass_idx, batch_size * n * sizeof(int));
cudaMalloc(&unass_cnt, batch_size * sizeof(int));
cudaMalloc(&unass_cnt_sum, batch_size * sizeof(int));
cudaMalloc(&cnt_tmp, batch_size * sizeof(int));
cudaMalloc(&max_idx, batch_size * n * sizeof(int));
cudaMalloc(&price, batch_size * n * sizeof(float));
cudaMalloc(&bid_increments, batch_size * n * sizeof(float));
cudaMalloc(&max_increments, batch_size * n * sizeof(float));

cudaMemsetAsync(out1_p, 0xff, batch_size * n * sizeof(int));
cudaMemsetAsync(assignment_inv, 0xff, batch_size * n * sizeof(int));
cudaMemsetAsync(price, 0, batch_size * n * sizeof(float));

int block_cnt = n / 1024;
float eps_val = {eps}f;
int n_iters = {iters};

for (int it = 0; it < n_iters; it++) {{
    emd_clear<<<1, batch_size>>>(batch_size, cnt_tmp, unass_cnt);
    emd_calc_unass_cnt<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, out1_p, unass_cnt);
    emd_calc_unass_cnt_sum<<<1, batch_size>>>(batch_size, unass_cnt, unass_cnt_sum);
    emd_calc_unass_idx<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, out1_p, unass_idx,
                                                                   unass_cnt, unass_cnt_sum, cnt_tmp);
    emd_Bid<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, in0_p, in1_p, eps_val, out1_p,
                                                        assignment_inv, price, bid, bid_increments,
                                                        max_increments, unass_cnt, unass_cnt_sum, unass_idx);
    emd_GetMax<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, out1_p, bid, bid_increments,
                                                           max_increments, max_idx);
    emd_Assign<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, out1_p, assignment_inv, price,
                                                           bid, bid_increments, max_increments, max_idx,
                                                           it == n_iters - 1);
}}
emd_CalcDist<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, in0_p, in1_p, out0_p, out1_p);

cudaFree(assignment_inv);
cudaFree(bid);
cudaFree(unass_idx);
cudaFree(unass_cnt);
cudaFree(unass_cnt_sum);
cudaFree(cnt_tmp);
cudaFree(max_idx);
cudaFree(price);
cudaFree(bid_increments);
cudaFree(max_increments);

cudaError_t err = cudaGetLastError();
ASSERT(err == cudaSuccess) << cudaGetErrorString(err);
"""


_BACKWARD_CUDA_SRC = """
int batch_size = in0->shape[0];
int n = in0->shape[1];
int block_cnt = n / 1024;

cudaMemsetAsync(out0_p, 0, batch_size * n * 3 * sizeof(float));
emd_grad_kernel<<<dim3(batch_size, block_cnt, 1), 1024>>>(batch_size, n, in0_p, in1_p, in2_p, in3_p, out0_p);

cudaError_t err = cudaGetLastError();
ASSERT(err == cudaSuccess) << cudaGetErrorString(err);
"""


class EMDFunction(Function):
    """auction-EMD 的 Jittor Function 包装。仅对 xyz1 求梯度（对齐原 PyTorch 实现）。"""

    def execute(self, xyz1: jt.Var, xyz2: jt.Var, eps: float = 0.005, iters: int = 50):
        assert xyz1.shape == xyz2.shape, f"xyz1/xyz2 shape mismatch: {xyz1.shape} vs {xyz2.shape}"
        B, N, C = xyz1.shape
        assert C == 3, f"expected last dim=3, got {C}"

        xyz1 = xyz1.float32()
        xyz2 = xyz2.float32()

        dist, assignment = jt.code(
            [(B, N), (B, N)], ["float32", "int32"], [xyz1, xyz2],
            cuda_header=_CUDA_HEADER,
            cuda_src=_forward_cuda_src(eps, iters),
        )
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        self.assignment = assignment
        return dist, assignment

    def grad(self, grad_dist: jt.Var, grad_assignment):
        # assignment（int32 索引）不可微，对应梯度恒为 None。
        xyz1, xyz2, assignment = self.xyz1, self.xyz2, self.assignment
        B, N, C = xyz1.shape
        grad_dist = grad_dist.float32()

        grad_xyz1 = jt.code(
            (B, N, C), "float32", [xyz1, xyz2, grad_dist, assignment],
            cuda_header=_CUDA_HEADER,
            cuda_src=_BACKWARD_CUDA_SRC,
        )
        return grad_xyz1, None


def earth_mover_distance(xyz1: jt.Var, xyz2: jt.Var, eps: float = 0.005, iters: int = 50):
    """auction-EMD 近似距离。

    Args:
        xyz1: (B, N, 3) 预测点云（有梯度）
        xyz2: (B, N, 3) GT 点云（无梯度），N 必须与 xyz1 相同且是 1024 的倍数
        eps: auction 算法步长参数（原实现默认 0.005）
        iters: auction 迭代次数（原实现默认 50）

    Returns:
        dist: (B, N) 每点平方距离（sqrt 后才是 L2 距离）
        assignment: (B, N) int32，匹配到 xyz2 的下标（近似双射，非精确）
    """
    return EMDFunction.apply(xyz1, xyz2, eps, iters)
