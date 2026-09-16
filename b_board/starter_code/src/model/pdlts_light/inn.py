"""PDLTS Light 的 INN 组件（Jittor 移植与轻量可逆实现）。

来源:
    - 上游 PyTorch 参考实现/models/layers/act_norm.py  (ActNormNd/1d/2d)
    - 上游 PyTorch 参考实现/models/layers/glow.py      (InvertibleLinear)
    - 上游 PyTorch 参考实现/models/layers/container.py (SequentialFlow)
    - 上游 PyTorch 参考实现/models/model_light/deflow.py (FlowAssembly)

轻量可逆实现的设计边界：
    原 chain: [iMonotoneBlock, ActNorm, iMonotoneBlock(preact), ActNorm]
    当前实现:  [ActNorm1d, AffineCoupling(even), ActNorm1d, AffineCoupling(odd)]

    仿射耦合层直接求逆，不依赖不动点迭代或 Lipschitz 约束。
    AffineCoupling 严格可逆（闭式），ActNorm1d 严格可逆（(x+b)*exp(w)），
    正逆向误差仍受浮点精度和缩放范围影响。

公共接口:
    module.forward(x)  或 module(x, logpx=None)     -> y 或 (y, logpx+ldj)
    module.inverse(y, logpy=None)                    -> x 或 (x, logpy-ldj)

输入约定: (B, N, C)，C = pc_channel + aug_channel，默认 3 + 48 = 51
"""

from typing import List, Optional

import numpy as np

import jittor as jt
from jittor import nn


# ============================================================================
# ActNorm1d: 首次 forward 用 batch 统计初始化，之后参数继续参与训练
# ============================================================================
class ActNorm1d(nn.Module):
    """Jittor 版 ActNorm for 3D 输入 (B, N, C)。

    forward:  y = (x + bias) * exp(weight)
    inverse:  x = y * exp(-weight) - bias
    logdet:   weight.sum()  (per-sample 常量)

    首次 forward 自动用 batch 统计初始化 weight / bias:
        bias    = -batch_mean   (over B*N axis)
        weight  = -0.5 * log(max(batch_var, 0.2))

    之后 is_inited=True，不再更新统计。

    对应原仓库: 上游 PyTorch 参考实现/models/layers/act_norm.py:9-86
    调整: shape 固定为 [1, 1, C] 以适配 (B, N, C) 布局（原仓库的 ActNorm1d
          shape=[1, -1] 假定通道在 dim=1，与 PD-LTS 点云布局不符）。
    """

    def __init__(self, num_features: int, eps: float = 1e-12):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 可学参数
        self.weight = jt.zeros((num_features,))
        self.bias = jt.zeros((num_features,))
        # 状态标志: 用 Python bool 管理, 不进 state_dict
        # 为了 save/load 正确, 用 Var 存储但 stop_grad
        self.is_inited = jt.zeros((1,))

    def _ensure_initialized(self, x: jt.Var) -> None:
        """首次 forward 时用 batch 统计填 weight / bias。"""
        if float(self.is_inited.item()) >= 0.5:
            return
        # x: (B, N, C). 跨 B*N 轴算 mean/var
        B, N, C = x.shape
        flat = x.detach().reshape(B * N, C)  # (BN, C)
        # 按通道 C 算
        batch_mean = flat.mean(dim=0)  # (C,)
        # 原仓库用 unbiased=False 的 var（pytorch 的 torch.var 默认是 unbiased=True
        # 但这里的初始化精度不敏感，走有偏估计）
        batch_var = ((flat - batch_mean.unsqueeze(0)) ** 2).mean(dim=0)  # (C,)
        # 数值保护
        batch_var = jt.maximum(batch_var, jt.array(0.2))

        self.bias.update(-batch_mean.detach())
        self.weight.update((-0.5 * jt.log(batch_var)).detach())
        self.is_inited.update(jt.ones((1,)))

    def execute(self, x: jt.Var, logpx: Optional[jt.Var] = None):
        self._ensure_initialized(x)
        # broadcast to (B, N, C)
        w = self.weight.view(1, 1, self.num_features)
        b = self.bias.view(1, 1, self.num_features)
        y = (x + b) * jt.exp(w)
        if logpx is None:
            return y
        ldj = self._logdetgrad(x)  # (B,)
        return y, logpx + ldj

    def inverse(self, y: jt.Var, logpy: Optional[jt.Var] = None):
        # inverse 前必须已 inited；若还没，报错比静默从 eval 去初始化更安全
        assert float(self.is_inited.item()) >= 0.5, \
            "ActNorm1d.inverse called before forward initialization"
        w = self.weight.view(1, 1, self.num_features)
        b = self.bias.view(1, 1, self.num_features)
        x = y * jt.exp(-w) - b
        if logpy is None:
            return x
        ldj = self._logdetgrad(y)
        return x, logpy - ldj

    def _logdetgrad(self, x: jt.Var) -> jt.Var:
        """per-sample logdet, shape (B,)。

        ActNorm1d 对每个点独立应用相同的仿射，所以
          log|det(dy/dx)| = sum_channels(weight) * N
        其中 N 是点数。
        """
        B, N, _ = x.shape
        return self.weight.sum() * N * jt.ones((B,))


# ============================================================================
# InvertibleLinear: QR init + W.inverse()
# ============================================================================
class InvertibleLinear(nn.Module):
    """y = x @ W^T；inverse 靠 W^{-1}；logdet = log|det W|。

    当前简化 FlowAssembly 方案**不使用**此层（只靠 ActNorm + AffineCoupling）。
    保留实现用于: 单测验证线性代数路径可用；后续若恢复 iMonotoneBlock，可作为 chain 的补充。

    对应原仓库: 上游 PyTorch 参考实现/models/layers/glow.py:10-41

    实现选择 (2026-05-08):
      原仓库写 `jt.linalg.einsum + W.inverse() + torch.det`; 在 Jittor+cupy 组合下
      `jt.linalg.einsum` 踩到 libcublas 版本不匹配错误 (cupy 要 libcublas.so.11,
      但 Jittor 自带 CUDA 12.2 只有 libcublas.so.12)。
      规避: 用 `jt.nn.linear(x, W)` 做主 linear, inv/det 走 numpy fallback (无 grad
      穿 det, 但 当前基础方案 InvertibleLinear 不进 chain, grad 穿 det 不是必须)。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # QR 初始化为正交阵, det = ±1, log|det| ≈ 0
        w_init = np.random.randn(dim, dim)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.W = jt.array(w_init)

    def execute(self, x: jt.Var, logpx: Optional[jt.Var] = None):
        # jt.nn.linear(x, W) 等价 x @ W.T (bias=None)
        # 对应原仓库 F.linear(x, self.W)
        y = nn.linear(x, self.W)
        if logpx is None:
            return y
        B, N, _ = x.shape
        logdet = self._logdetgrad()
        return y, logpx + logdet * float(N) * jt.ones((B,))

    def inverse(self, y: jt.Var, logpy: Optional[jt.Var] = None):
        # numpy fallback: W^{-1} (dim 很小，CPU numpy 足够快；无 grad 穿 inverse)
        W_inv_np = np.linalg.inv(self.W.numpy())
        W_inv = jt.array(W_inv_np.astype(np.float32))
        x = nn.linear(y, W_inv)
        if logpy is None:
            return x
        B, N, _ = y.shape
        logdet = self._logdetgrad()
        return x, logpy - logdet * float(N) * jt.ones((B,))

    def _logdetgrad(self) -> jt.Var:
        # log|det W|; numpy slogdet (无 grad 穿 det)
        _, logabsdet = np.linalg.slogdet(self.W.numpy())
        return jt.array(np.array(logabsdet, dtype=np.float32))


# ============================================================================
# AffineCoupling (RealNVP-style): 当前简化代替 iMonotoneBlock
# ============================================================================
class AffineCoupling(nn.Module):
    """RealNVP-style affine coupling on (B, N, C)。

    给定 binary mask m (shape [C]), 1 表示"保留", 0 表示"变换":
        y = m * x + (1 - m) * (x * exp(s(x_masked)) + t(x_masked))

    其中 s / t 由同一个 MLP 输出（head 拆成两半）。
    严格可逆:
        x_masked = m * y
        y_unm    = (1 - m) * y
        out      = MLP(x_masked) -> (log_s, t)
        log_s    = clamp(log_s * (1 - m)) * 0.1  # 值域收紧，避免 exp 爆
        x_unm    = (y_unm - (1 - m) * t) * exp(-log_s)
        x        = x_masked + x_unm

    logdet = sum_over_C( (1 - m) * log_s )  -> per-point
    然后对 N 求和得到 per-sample logdet。
    """

    def __init__(self, channel: int, hidden: int = 64, mask_parity: str = "even",
                 log_scale_clamp: float = 0.1):
        super().__init__()
        self.channel = channel
        self.mask_parity = mask_parity
        self.log_scale_clamp = log_scale_clamp

        # mask: (C,), float32, 0 or 1
        m = np.zeros(channel, dtype=np.float32)
        if mask_parity == "even":
            m[0::2] = 1.0
        elif mask_parity == "odd":
            m[1::2] = 1.0
        else:
            raise ValueError(f"unsupported mask_parity: {mask_parity}")
        # register 为 Var 但不加梯度
        self.mask = jt.array(m).stop_grad()

        # net: C -> hidden -> hidden -> 2*C (log_s, t)
        self.net = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel * 2),
        )
        # 最后一层初始化为 0: 初始时 log_s=0, t=0，等价 identity
        last = self.net[-1]
        _w = np.zeros((channel * 2, hidden), dtype=np.float32)
        _b = np.zeros((channel * 2,), dtype=np.float32)
        last.weight = jt.array(_w)
        last.bias = jt.array(_b)

    def _split_scale_shift(self, out: jt.Var):
        """out: (B, N, 2C) -> log_s, t (B, N, C)。"""
        C = self.channel
        log_s = out[..., :C]
        t = out[..., C:]
        # clamp log_s 到 [-log_scale_clamp, log_scale_clamp] 的等效软收紧
        # 原 RealNVP 用 tanh，这里用乘一个小常数保证数值稳定
        log_s = jt.tanh(log_s) * self.log_scale_clamp
        return log_s, t

    def execute(self, x: jt.Var, logpx: Optional[jt.Var] = None):
        # x: (B, N, C)
        m = self.mask.view(1, 1, -1)  # (1, 1, C)
        x_masked = x * m
        out = self.net(x_masked)
        log_s, t = self._split_scale_shift(out)
        # 只让被变换的通道（m==0）生效
        log_s = log_s * (1.0 - m)
        t = t * (1.0 - m)
        y = x_masked + (1.0 - m) * (x * jt.exp(log_s) + t)
        if logpx is None:
            return y
        # per-sample ldj = sum over (N, C) 的 log_s
        # Jittor 的 jt.sum 不支持 dim=tuple (只支持 dim=int 或 dims=NanoVector)
        ldj = log_s.sum(dims=[1, 2])  # (B,)
        return y, logpx + ldj

    def inverse(self, y: jt.Var, logpy: Optional[jt.Var] = None):
        m = self.mask.view(1, 1, -1)
        # 被保留的通道等于 x 同值
        x_masked = y * m
        out = self.net(x_masked)
        log_s, t = self._split_scale_shift(out)
        log_s = log_s * (1.0 - m)
        t = t * (1.0 - m)
        # 从 y_unm 反解 x_unm:
        # y_unm = (1 - m) * (x * exp(log_s) + t)
        # x_unm = (y * (1 - m) - t * (1 - m)) * exp(-log_s)
        x = x_masked + (1.0 - m) * ((y - t) * jt.exp(-log_s))
        if logpy is None:
            return x
        ldj = log_s.sum(dims=[1, 2])
        return x, logpy - ldj


# ============================================================================
# SequentialFlow: 正向按顺序 forward, 反向按逆序 inverse
# ============================================================================
class SequentialFlow(nn.Module):
    """对应原仓库 上游 PyTorch 参考实现/models/layers/container.py:4-31。"""

    def __init__(self, layers_list: List[nn.Module]):
        super().__init__()
        self.chain = nn.ModuleList(layers_list)

    def execute(self, x: jt.Var, logpx: Optional[jt.Var] = None):
        if logpx is None:
            for layer in self.chain:
                x = layer(x)
            return x
        for layer in self.chain:
            x, logpx = layer(x, logpx)
        return x, logpx

    def inverse(self, y: jt.Var, logpy: Optional[jt.Var] = None):
        if logpy is None:
            for layer in reversed(self.chain):
                y = layer.inverse(y)
            return y
        for layer in reversed(self.chain):
            y, logpy = layer.inverse(y, logpy)
        return y, logpy


# ============================================================================
# FlowAssembly: 一个 INN 子模块，chain 在 当前简化为全 AffineCoupling + ActNorm
# ============================================================================
class FlowAssembly(SequentialFlow):
    """对应原仓库 上游 PyTorch 参考实现/models/model_light/deflow.py:273-343。

    原版:
        [iMonotoneBlock, ActNorm, iMonotoneBlock(preact), ActNorm]
    简化后:
        [ActNorm1d, AffineCoupling(even), ActNorm1d, AffineCoupling(odd)]
    """

    def __init__(self, channel: int, hidden: int = 64,
                 log_scale_clamp: float = 0.1):
        chain = [
            ActNorm1d(channel),
            AffineCoupling(channel, hidden=hidden, mask_parity="even",
                           log_scale_clamp=log_scale_clamp),
            ActNorm1d(channel),
            AffineCoupling(channel, hidden=hidden, mask_parity="odd",
                           log_scale_clamp=log_scale_clamp),
        ]
        super().__init__(chain)
