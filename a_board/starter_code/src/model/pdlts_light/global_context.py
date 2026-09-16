"""全局上下文模块（轻量 Hilbert-transformer）。

在 MLGC 局部特征 (aug) 之后、INN 之前插入一个 2-layer full self-attention block，
通过 Hilbert 序列化提供 patch-global 感受野。

架构:
    aug (B, N, feature_dim)
      → Hilbert reorder (按 xyz 坐标)
      → + learned 1D position embedding
      → 2× Pre-LN TransformerLayer (MHA + FFN)
      → project back to feature_dim
      → restore original order
      → residual: aug + gamma * global_block(aug)

诊断输出（挂在 self._last_diag 上）:
    global_gamma, global_delta_norm, feature_mlgc_norm,
    global_delta_ratio, attention_entropy_mean, gamma_grad_available

配置与作用范围:
    - 只改 aug 特征 (MLGC 输出到 INN 输入之间)，不动 loss / K / INN / FBM / cut_channel。
    - 默认关闭 (global_context_mode="off")。
    - gamma 初始化为 0，保证 base 行为完全一致。
"""

from typing import Dict, Optional

import numpy as np

import jittor as jt
from jittor import nn

from .hilbert import hilbert_sort_indices


# ---------------------------------------------------------------------------
# 自定义 LayerNorm（兼容 Jittor 无内置 LayerNorm 的情况）
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """Per-feature LayerNorm。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = jt.ones((normalized_shape,))
        self.bias = jt.zeros((normalized_shape,))

    def execute(self, x: jt.Var) -> jt.Var:
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
        rstd = jt.sqrt(var + self.eps)
        return (x - mean) / rstd * self.weight + self.bias


# ---------------------------------------------------------------------------
# Transformer Layer (Pre-LN, full self-attention)
# ---------------------------------------------------------------------------

class _TransformerLayer(nn.Module):
    """Pre-LN Transformer block: MHA + FFN, with attention entropy logging."""

    def __init__(self, d_model: int, n_heads: int, ffn_multiplier: int = 2):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = float(self.d_head) ** 0.5

        # Pre-LN for attention
        self.ln1 = _LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-LN for FFN
        self.ln2 = _LayerNorm(d_model)
        ffn_dim = d_model * ffn_multiplier
        self.ffn1 = nn.Linear(d_model, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, d_model)

    def execute(self, x: jt.Var) -> jt.Var:
        """Forward + return attention entropy (float, detached)."""
        B, N, D = x.shape

        # ---- Multi-Head Self-Attention ----
        residual = x
        x_norm = self.ln1(x)

        q = self.q_proj(x_norm).reshape(B, N, self.n_heads, self.d_head)
        k = self.k_proj(x_norm).reshape(B, N, self.n_heads, self.d_head)
        v = self.v_proj(x_norm).reshape(B, N, self.n_heads, self.d_head)

        # (B, N, h, d) → (B, h, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = q @ k.transpose(0, 1, 3, 2)  # (B, h, N, N)
        attn_scores = attn_scores / self.scale
        attn_weights = nn.softmax(attn_scores, dim=-1)

        # Entropy: per-head mean entropy over query positions
        # (B, h, N, N) → mean over heads and queries → scalar
        entropy = -(attn_weights * jt.log(attn_weights + 1e-10)).sum(dim=-1).mean()

        attn_out = attn_weights @ v  # (B, h, N, d_head)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out

        # ---- Feed-Forward Network ----
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn2(nn.relu(self.ffn1(x_norm)))
        x = residual + ffn_out

        return x, entropy


# ---------------------------------------------------------------------------
# Global Context Block
# ---------------------------------------------------------------------------

class GlobalContextBlock(nn.Module):
    """轻量 Hilbert-transformer global context block。

    单变量约束:
        - 只修改 aug 特征: aug_new = aug + gamma * global_block(aug, xyz)
        - gamma 初始化为 0，保证 base 行为完全一致。
        - xyz 只用于 Hilbert 排序，不参与 attention 计算。
    """

    def __init__(
        self,
        feature_dim: int = 48,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_multiplier: int = 2,
        max_patch_size: int = 1024,
        hilbert_bits: int = 10,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hilbert_bits = hilbert_bits

        # 可学习残差门控，初始化为 0 → base 行为完全一致
        self.gamma = jt.zeros((1,))

        # 学习 1D 位置嵌入
        self.pos_emb = nn.Embedding(max_patch_size, d_model)

        # 维度投影
        self.in_proj = nn.Linear(feature_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, feature_dim, bias=False)

        # Transformer 层
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(_TransformerLayer(d_model, n_heads, ffn_multiplier))

        # 诊断暂存
        self._last_diag: Dict[str, float] = {}

    def execute(self, aug: jt.Var, xyz: jt.Var) -> jt.Var:
        """Forward pass。

        Args:
            aug: (B, N, feature_dim)  MLGC unit_coupling 输出
            xyz: (B, N, 3)            noisy patch 坐标 (已中心化)

        Returns:
            aug_global: (B, N, feature_dim)  残差项 (由外部乘 gamma 后加回 aug)
        """
        B, N, _ = aug.shape

        # 1. Hilbert 排序索引 (从 xyz 坐标, detached numpy)
        xyz_np = np.asarray(xyz.stop_grad().numpy())
        sort_idx_np, unsort_idx_np = hilbert_sort_indices(xyz_np, bits=self.hilbert_bits)
        sort_idx = jt.array(sort_idx_np)
        unsort_idx = jt.array(unsort_idx_np)

        # 批次索引
        bi = jt.arange(B).unsqueeze(-1).broadcast([B, N])

        # 2. 按 Hilbert 顺序重排特征
        aug_ordered = aug[bi, sort_idx]  # (B, N, feature_dim)

        # 3. 投影 + 位置嵌入
        x = self.in_proj(aug_ordered)  # (B, N, d_model)
        pos = self.pos_emb(jt.arange(N))
        x = x + pos.unsqueeze(0)

        # 4. Transformer 层
        entropies = []
        for layer in self.layers:
            x, ent = layer(x)
            entropies.append(float(ent.item()) if ent is not None else 0.0)

        # 5. 投影回原始维度
        out = self.out_proj(x)  # (B, N, feature_dim)

        # 6. 恢复原始顺序
        out_restored = out[bi, unsort_idx]  # (B, N, feature_dim)

        # 7. 诊断记录
        delta_norm = jt.sqrt((out_restored ** 2).sum(dim=-1)).mean()
        feat_norm = jt.sqrt((aug ** 2).sum(dim=-1)).mean()

        self._last_diag = {
            "global_gamma": float(self.gamma.item()),
            "global_delta_norm": float(delta_norm.item()),
            "feature_mlgc_norm": float(feat_norm.item()),
            "global_delta_ratio": float(
                (delta_norm / (feat_norm + 1e-8)).item()
            ),
            "global_applied_delta_ratio": float(
                ((jt.abs(self.gamma) * delta_norm) / (feat_norm + 1e-8)).item()
            ),
            "attention_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        }

        return out_restored

    def last_diag(self) -> Dict[str, float]:
        return dict(self._last_diag)
