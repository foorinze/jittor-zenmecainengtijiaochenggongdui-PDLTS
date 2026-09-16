"""失败对照：auction-EMD 最小可行 Jittor 移植（research_only，仅探针使用）。

源算法：上游 PyTorch 参考实现/metric/emd/{emd.cpp,emd_cuda.cu}
（Minghua Liu 的 auction algorithm 近似 EMD，已在上游 PyTorch 侧验证
forward/backward 数值正确、runtime 8.8~9.9ms/patch）。

本包只做"移植可行性"探针，不是生产级优化实现：
    - 用 Jittor `jt.code` 内联 CUDA（vs 编译扩展注册），因为 `jt.code` 是
      Jittor 官方文档推荐路径，且不需要额外的 setup.py/pybind11 构建链，
      直接复用 Jittor 自身的 JIT 编译器（nvcc 调用方式与 Jittor 内部算子
      完全一致，不存在"两套 JIT 冲突"——这正是 D1 Q4 要验证/排除的风险）。
    - kernel 逐条从原 CUDA 源码搬运（不改算法逻辑），只把 PyTorch 侧由
      Python 预分配 + 传入的中间状态数组（price/bid/assignment_inv/...）
      改为在 `cuda_src` 内部 `cudaMalloc`/`cudaFree` 自管理（Jittor 版更
      简单，不需要在 Python 侧构造 10 个輔助 tensor）。
    - batch_size <= 512、n % 1024 == 0 的约束沿用原实现（`calc_unass_cnt_sum`
      的共享内存数组硬编码 512）。
"""

from .emd_op import earth_mover_distance, EMDFunction

__all__ = ["earth_mover_distance", "EMDFunction"]
