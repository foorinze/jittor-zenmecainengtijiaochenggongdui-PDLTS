"""MLGC 组件（PDLTS Light 的特征提取部分，Jittor 移植）

来源: PD-LTS 原始实现 models/model_light/layer.py（211 行）。


主要类:
    FullyConnectedLayer  - Linear + 激活
    noiseEdgeConv        - unit_coupling 用的 EdgeConv 变体
    PreConv              - 进入主干前的 EdgeConv (Conv2d 风格)
    EdgeConv             - 主干 MLGC 层 (Conv2d 风格；与 starter_code/src/model/feature.py:6 同名但不同结构)
    FeatMergeUnit        - EdgeConv 输出到注入特征的适配 (Conv1d + BN)

Helper:
    knn_group(x, i)      - 按 knn idx gather feature
    safe_knn(u, k, K)    - jt.misc.knn 的安全副本（越界守卫），主流程统一用它
    get_knn_idx(k, f, q) - brute-force knn (fallback/单测对照，非主流程)
"""

from typing import Optional

import numpy as np

import jittor as jt
from jittor import nn


def knn_group(x: jt.Var, i: jt.Var) -> jt.Var:
    """按 knn idx gather feature。

    Args:
        x: (B, N, C)
        i: (B, M, k)  每个 query point 的 k 个邻居在 x 中的 index

    Returns:
        y: (B, M, k, C)

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:13-30
    """
    B, N, C = x.shape
    _, M, k = i.shape
    # batch 索引 (B, 1)
    idxb = jt.arange(B).view(-1, 1)
    # flatten k 维度后再展开: (B, M*k) -> gather -> (B, M, k, C)
    flat_i = i.reshape(B, M * k)
    y = x[idxb, flat_i].view(B, M, k, C)
    return y


def safe_knn(unknown: jt.Var, known: jt.Var, k: int):
    """jt.misc.knn 的仓库内安全副本（逐数值等价，仅多一个越界守卫）。

    背景：jittor 1.3.10 的 `jt.misc.knn`（misc.py:84）自定义内核按 `index`
    直接寻址 `dist2/idx[j*K+i]`，缺少 `j >= n` 的守卫；其
    auto_parallel(2, block_num=256) 在 b*n 不能被 256 整除时会产生越界线程，
    向输出缓冲之后的显存写入（实测：整云推理阶段 b*n=586、K=1024 时约
    182 个越界线程 × 8KB ≈ 1.4MB），腐蚀相邻的显存分配（典型受害者是模型内
    edge-conv 的 knn_idx），随后 gather 触发 cudaErrorIllegalAddress(700)，
    表现为推理进程崩溃。是否触发高度依赖进程的显存布局，因此同一份代码在
    不同 checkpoint 上时崩时不崩，排查时容易误判为数据问题。

    本副本只加一行守卫，内核逻辑、并行方式与数值结果都与原版完全一致。

    Args:
        unknown (var): shape [b, n, c]，仅支持 c=3（与原版一致）
        known (var): shape [b, m, c]
        k (int): 近邻数

    Returns:
        (dists2, idx): 与 jt.misc.knn 相同。
    """
    b, n, c = unknown.shape
    _, m, _ = known.shape
    dists2 = jt.empty((b, n, k), dtype="float")
    idx = jt.empty((b, n, k), dtype="int")
    src = '''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void knn_kernel(int b, int batch_index, int n, int index, int m,
                        const float *__restrict__ unknown,
                        const float *__restrict__ known,
                        float *__restrict__ dist2,
                        int *__restrict__ idx) {

#define K %s
    // 守卫：auto_parallel 尾块的越界线程直接返回，禁止越界写入。
    if (index >= n) return;
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * K;
    idx += batch_index * n * K;
    int j = index;
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        float tmp_dist[K];
        int tmp_idx[K];
        #pragma unroll
        for (int i=0; i<K; i++) tmp_dist[i] = 1e30;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

            int first = -1;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (first == -1 && d<tmp_dist[i])
                    first = i;
            if (first == -1) continue;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (K-1-i > first) {
                    tmp_dist[K-1-i] = tmp_dist[K-2-i];
                    tmp_idx[K-1-i] = tmp_idx[K-2-i];
                }
            tmp_dist[first] = d;
            tmp_idx[first] = k;
        }
        #pragma unroll
        for (int i=0; i<K; i++) {
            dist2[j * K + i] = tmp_dist[i];
            idx[j * K + i] = tmp_idx[i];
        }
    }
}
    knn_kernel(in0->shape[0], 0, in0->shape[1], 0, in1->shape[1], in0_p, in1_p, out0_p, out1_p);
    ''' % k
    return jt.code([unknown, known], [dists2, idx],
                   cpu_src=src,
                   cuda_src=src)


def get_knn_idx(k: int, f: jt.Var, q: Optional[jt.Var] = None, offset: int = 0) -> jt.Var:
    """Brute-force 欧式距离 KNN (返回 index, 不含自身时用 offset=1)。

    主流程优先用 jt.misc.knn；本函数作为 fallback 和单测对照。

    Args:
        k: 邻居数
        f: (B, N, C) reference points
        q: (B, M, C) query points；None 时 q = f
        offset: 跳过前 offset 个最近邻（通常 offset=1 排除自身）

    Returns:
        idx: (B, M, k)

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:33-56
    """
    if q is None:
        q = f
    B, N, C = f.shape
    _, M, _ = q.shape
    # (B, M, N, C) - (B, M, N, C)  via broadcast
    _f = f.unsqueeze(1).broadcast([B, M, N, C])
    _q = q.unsqueeze(2).broadcast([B, M, N, C])
    dist = ((_f - _q) ** 2).sum(dim=3)  # (B, M, N)
    # 升序 topk, 跳过前 offset 个（通常 offset=1 排除自身）
    K = k + offset
    # jt.topk 返回 (values, idx)，按 starter_code/src/model/feature.py:195 的用法验证
    _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[..., offset : K]


class FullyConnectedLayer(nn.Module):
    """Linear + 激活（ReLU / ELU / LeakyReLU / None）。

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:60-79
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: Optional[str] = None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def execute(self, x: jt.Var) -> jt.Var:
        return self.activation(self.linear(x))


class noiseEdgeConv(nn.Module):
    """unit_coupling 专用 EdgeConv（Linear 风格，非 Conv2d）。

    输入 xyz (B, N, 3)，输出 aug feature (B, N, out_channel)。

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:83-116
    """

    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int,
                 bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(in_channel * 2, hidden_channel, bias=bias)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear3 = nn.Linear(in_channel, hidden_channel, bias=bias)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear5 = nn.Linear(hidden_channel, out_channel, bias=bias)
        # 原仓库最后一层初始化: weight ~ N(0, 0.05), bias = 0
        # 用 numpy 生成再 assign，避免依赖 jt.init 具体 API
        _w = np.random.randn(out_channel, hidden_channel).astype(np.float32) * 0.05
        self.linear5.weight = jt.array(_w)
        self.linear5.bias = jt.zeros((out_channel,))

    def execute(self, f: jt.Var, knn_idx: jt.Var) -> jt.Var:
        """
        Args:
            f: (B, N, C)
            knn_idx: (B, N, k)
        Returns:
            (B, N, out_channel)
        """
        # gather 邻居特征
        knn_feat = knn_group(f, knn_idx)  # (B, N, k, C)
        B, N, k, C = knn_feat.shape
        f_tiled = f.unsqueeze(2).broadcast([B, N, k, C])
        # 拼接 [邻居, 邻居-自身]，与原仓库一致（注释写 "C*3" 其实是 2*C）
        x = jt.concat([knn_feat, knn_feat - f_tiled], dim=-1)  # (B, N, k, 2C)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        # (B, N, k, h) -> (B, N, h) max over k
        x = jt.max(x, dim=2)
        # 自身特征的分支
        f = nn.relu(self.linear3(f))
        f = nn.relu(self.linear4(f))
        x = x + f
        x = self.linear5(x)
        return x


class PreConv(nn.Module):
    """进入主干前的 EdgeConv（Conv2d + BN + LeakyReLU 风格）。

    输入 xyz (B, N, 3)，输出 (B, N, out_channel)。

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:118-142
    """

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        in_channel2 = in_channel * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel2, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        )

    def execute(self, f: jt.Var, knn_idx: jt.Var) -> jt.Var:
        """
        Args:
            f: (B, N, C)
            knn_idx: (B, N, k)
        Returns:
            (B, N, out_channel)
        """
        knn_feat = knn_group(f, knn_idx)  # (B, N, k, C)
        B, N, k, C = knn_feat.shape
        f_tiled = f.unsqueeze(2).broadcast([B, N, k, C])
        # [自身, 邻居-自身]
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)  # (B, N, k, 2C)
        # 转为 Conv2d 期望的布局 (B, 2C, N, k)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        # max over k -> (B, out, N) -> (B, N, out)
        x = jt.max(x, dim=-1)
        x = x.transpose(1, 2)
        return x


class EdgeConv(nn.Module):
    """主干 MLGC 层（Conv2d + BN + LeakyReLU 风格）。concat=True 时 concat 自身特征。

    警告: 这个类与 starter_code/src/model/feature.py:6 的 EdgeConv **同名但不同**。
    PDLTS 用的是这一版（Conv2d 风格），不要混用。

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:144-186
    """

    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int,
                 concat: bool = True):
        super().__init__()
        self.concat = concat
        # concat=False 时原仓库把 hidden_channel 加 32，保持连接处通道数
        if not concat:
            hidden_channel = hidden_channel + 32
        self.convs = nn.ModuleList()
        conv_first = nn.Sequential(
            nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=1),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.05),
        )
        self.convs.append(conv_first)
        conv_second = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        )
        self.convs.append(conv_second)

    def execute(self, f: jt.Var, knn_idx: jt.Var) -> jt.Var:
        """
        Args:
            f: (B, N, C)
            knn_idx: (B, N, k)
        Returns:
            concat=True:  (B, N, out_channel + C)
            concat=False: (B, N, out_channel)
        """
        knn_feat = knn_group(f, knn_idx)  # (B, N, k, C)
        B, N, k, C = knn_feat.shape
        f_tiled = f.unsqueeze(2).broadcast([B, N, k, C])
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)  # (B, N, k, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, N, k)
        for conv in self.convs:
            x = conv(x)
        x = jt.max(x, dim=-1)  # (B, out, N)
        x = x.transpose(1, 2)  # (B, N, out)
        if self.concat:
            x = jt.concat([x, f], dim=-1)
        return x


class FeatMergeUnit(nn.Module):
    """EdgeConv 输出到注入特征的适配（Conv1d + BN + ReLU）。

    对应原仓库: PD-LTS 原始实现 models/model_light/layer.py:188-210
    """

    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        ))

    def execute(self, x: jt.Var) -> jt.Var:
        """
        Args:
            x: (B, N, C_in)
        Returns:
            (B, N, C_out)
        """
        # (B, N, C) -> (B, C, N) 供 Conv1d 用
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        return x
