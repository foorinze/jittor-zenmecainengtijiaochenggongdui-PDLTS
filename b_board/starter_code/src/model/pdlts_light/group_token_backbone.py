"""GroupToken-HilbertAttention 骨干网络。

PointMamba-inspired full backbone replacement for MLGC:
  xyz (B,N,3)
    → FPS centers (B,G,3) + KNN groups (B,G,S,3)
    → LocalEncoder → tokens (B,G,C)
    → center_pos_mlp → pos (B,G,C)
    → Hilbert + Hilbert-trans bidirectional serialization
    → TokenMixer (depth×self-attention)
    → CrossAttentionUpsample → point_features (B,N,C)
    → AugHead + 12×InjHead
    → aug (B,N,48), inj_f[0..11] (B,N,51)

结构实验见 experiments/a_board_atlas.md。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import jittor as jt
from jittor import nn

from .hilbert import hilbert_sort_indices
from .layer import safe_knn


# ---------------------------------------------------------------------------
# FPS (greedy farthest point sampling)
# ---------------------------------------------------------------------------

def _sample_centers(xyz: jt.Var, n_centers: int) -> Tuple[jt.Var, jt.Var]:
    """Uniform random sampling of centers (numpy-based, O(1) graph ops)。

    替代原贪心 FPS，避免 Jittor graph 节点累积导致长训 segfault。
    采样在 numpy 中完成，只将最终 index 传入 Jittor graph 做 gather。

    Args:
        xyz: (B, N, 3)
        n_centers: number of sampled centers

    Returns:
        centers_xyz: (B, n_centers, 3)
        center_idx:  (B, n_centers) int
    """
    B, N, _ = xyz.shape
    # Random sampling in numpy (detached, no graph accumulation)
    xyz_np = np.asarray(xyz.stop_grad().numpy())
    center_idx_list = []
    for b in range(B):
        idx = np.random.choice(N, size=n_centers, replace=False).astype(np.int64)
        center_idx_list.append(idx)
    center_idx = jt.array(np.stack(center_idx_list, axis=0))  # (B, n_centers)
    bi = jt.arange(B).unsqueeze(-1)                            # (B, 1)
    centers_xyz = xyz[bi, center_idx]                          # (B, n_centers, 3)
    return centers_xyz, center_idx


# ---------------------------------------------------------------------------
# Local Group Encoder (PointNet-style)
# ---------------------------------------------------------------------------

def _norm1d(c: int, kind: str):
    """1D 归一化层工厂。batch=BatchNorm1d(GroupToken 默认，train/eval 行为不同，batch=1 易坏)；
    group=GroupNorm(batch 无关，train=eval 一致，displacement head 用)。"""
    if kind == "batch":
        return nn.BatchNorm1d(c)
    if kind == "group":
        g = 8 if c % 8 == 0 else (4 if c % 4 == 0 else 1)
        return nn.GroupNorm(g, c)
    raise ValueError(f"unknown norm kind: {kind}")


class _LocalEncoder(nn.Module):
    """Per-group PointNet encoder。

    Input:  (B, G, S, 3) — S points per group, centered by group center
    Output: (B, G, C)     — one token per group
    """

    def __init__(self, out_dim: int = 128, norm: str = "batch"):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), _norm1d(64, norm), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), _norm1d(128, norm), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, out_dim, 1), _norm1d(out_dim, norm), nn.ReLU())

    def execute(self, groups: jt.Var) -> jt.Var:
        """
        Args:
            groups: (B, G, S, 3) — coordinates, centered by group center
        Returns:
            tokens: (B, G, C)
        """
        B, G, S, _ = groups.shape
        x = groups.reshape(B * G, S, 3)        # (B*G, S, 3)
        x = x.transpose(1, 2)                   # (B*G, 3, S)
        x = self.conv1(x)                       # (B*G, 64, S)
        x = self.conv2(x)                       # (B*G, 128, S)
        x = self.conv3(x)                       # (B*G, C, S)
        x = jt.max(x, dim=-1)                   # (B*G, C)
        return x.reshape(B, G, -1)


# ---------------------------------------------------------------------------
# PointIdentity MLP — preserves per-point info bypassing the group bottleneck
# ---------------------------------------------------------------------------

class _PointIdentityMLP(nn.Module):
    """Lightweight per-point MLP on raw coordinates.

    Preserves point-level geometric identity that MaxPool in the group encoder
    would otherwise discard. Outputs (B, N, C_id) — small feature dim to keep
    the path from dominating the distribution branch.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def execute(self, xyz: jt.Var) -> jt.Var:
        """xyz: (B, N, 3) → (B, N, out_dim)"""
        return self.mlp(xyz)


# ---------------------------------------------------------------------------
# Token Mixer Layer (Pre-LN self-attention)
# ---------------------------------------------------------------------------

class _TokenMixerLayer(nn.Module):
    """Pre-LN self-attention + FFN block for token-level mixing."""

    def __init__(self, dim: int, heads: int = 4, ffn_mult: int = 2):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d_head = dim // heads
        self.scale = float(self.d_head) ** 0.5

        # Pre-LN attention
        self.ln1_weight = jt.ones((dim,))
        self.ln1_bias = jt.zeros((dim,))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Pre-LN FFN
        self.ln2_weight = jt.ones((dim,))
        self.ln2_bias = jt.zeros((dim,))
        ffn_dim = dim * ffn_mult
        self.ffn1 = nn.Linear(dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, dim)

    def _layer_norm(self, x: jt.Var, w: jt.Var, b: jt.Var, eps: float = 1e-5) -> jt.Var:
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
        return (x - mean) / jt.sqrt(var + eps) * w + b

    def execute(self, x: jt.Var) -> jt.Var:
        B, L, D = x.shape

        # --- Self-Attention ---
        residual = x
        x_norm = self._layer_norm(x, self.ln1_weight, self.ln1_bias)

        q = self.q_proj(x_norm).reshape(B, L, self.heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, L, self.heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, L, self.heads, self.d_head).permute(0, 2, 1, 3)

        attn = q @ k.transpose(0, 1, 3, 2) / self.scale  # (B, heads, L, L)
        attn_w = nn.softmax(attn, dim=-1)

        # entropy diagnostic
        entropy = -(attn_w * jt.log(attn_w + 1e-10)).sum(dim=-1).mean()

        out = attn_w @ v  # (B, heads, L, d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.o_proj(out)
        x = residual + out

        # --- FFN ---
        residual = x
        x_norm = self._layer_norm(x, self.ln2_weight, self.ln2_bias)
        x = residual + self.ffn2(nn.relu(self.ffn1(x_norm)))

        return x, float(entropy.item())


class _TokenMixer(nn.Module):
    """Stack of _TokenMixerLayer with Hilbert serialization."""

    def __init__(self, dim: int, depth: int = 4, heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            _TokenMixerLayer(dim, heads) for _ in range(depth)
        ])

    def execute(self, x: jt.Var) -> jt.Var:
        entropies = []
        for layer in self.layers:
            x, ent = layer(x)
            entropies.append(ent)
        return x, entropies


# ---------------------------------------------------------------------------
# Cross-Attention Upsample: token → per-point
# ---------------------------------------------------------------------------

class _CrossAttentionUpsample(nn.Module):
    """Learnable token-to-point upsampling via cross-attention.

    Each point i attends to its k nearest tokens (by spatial distance),
    producing a weighted combination of token features.
    """

    def __init__(self, dim: int = 128, k: int = 8, knn_backend: str = "numpy"):
        super().__init__()
        self.dim = dim
        self.k = k
        self.knn_backend = knn_backend  # "numpy"(GroupToken 默认) | "jittor"(GPU jt.misc.knn)
        self.scale = float(dim) ** 0.5

        # Point query: xyz → query vector
        self.point_query = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
        # Token key/value projections
        self.token_key = nn.Linear(dim, dim, bias=False)
        self.token_value = nn.Linear(dim, dim, bias=False)

    def execute(self, tokens: jt.Var, token_centers: jt.Var, xyz: jt.Var) -> jt.Var:
        """
        Args:
            tokens:  (B, T, C)  T=2G after bidirectional serialization
            token_centers: (B, T, 3) — same T centers (broadcast for 2 directions)
            xyz:     (B, N, 3)  original patch points

        Returns:
            point_feat: (B, N, C)
        """
        B, T, C = tokens.shape
        _, N, _ = xyz.shape

        # Token K/V
        tk = self.token_key(tokens)   # (B, T, C)
        tv = self.token_value(tokens)  # (B, T, C)

        # Point Q from xyz
        pq = self.point_query(xyz)     # (B, N, C)

        # Find k nearest tokens for each point (by spatial distance to centers)
        # token_centers may only have G unique centers; repeat for bidirectional
        if self.knn_backend == "jittor":
            # GPU safe_knn（jt.misc.knn 带越界守卫副本）：query=xyz, ref=token_centers，
            # 取每点最近 k 个 token。与 numpy 分支语义等价（同 top-k 最近 token 索引），
            # 但全程在 GPU、无 host 搬运。
            _, knn_idx = safe_knn(xyz, token_centers, self.k)  # (B, N, k) -> token index
            knn_idx = knn_idx.int32()
        else:
            # Use numpy KNN (detached) to find nearest centers (GroupToken 默认路径)
            centers_np = np.asarray(token_centers.stop_grad().numpy())  # (B, T, 3)
            xyz_np = np.asarray(xyz.stop_grad().numpy())                # (B, N, 3)

            knn_idx_list = []
            for b in range(B):
                diff = xyz_np[b, :, None, :] - centers_np[b, None, :, :]   # (N, T, 3)
                dist2 = (diff ** 2).sum(axis=-1)                             # (N, T)
                top_k = np.argsort(dist2, axis=-1)[:, :self.k].astype(np.int64)  # (N, k)
                knn_idx_list.append(top_k)
            knn_idx = jt.array(np.stack(knn_idx_list, axis=0))  # (B, N, k)

        # Gather token K/V for each point's nearest tokens
        bi = jt.arange(B).unsqueeze(-1).unsqueeze(-1).broadcast([B, N, self.k])
        tk_local = tk[bi, knn_idx]  # (B, N, k, C)
        tv_local = tv[bi, knn_idx]  # (B, N, k, C)

        # Cross-attention: Q(point) × K(tokens) → weighted V
        attn_scores = (pq.unsqueeze(2) * tk_local).sum(dim=-1) / self.scale  # (B, N, k)
        attn_w = nn.softmax(attn_scores, dim=-1)                              # (B, N, k)

        # Weighted sum
        point_feat = (attn_w.unsqueeze(-1) * tv_local).sum(dim=2)  # (B, N, C)

        # Diagnostics
        entropy = -(attn_w * jt.log(attn_w + 1e-10)).sum(dim=-1).mean()

        return point_feat, float(entropy.item()), knn_idx


# ---------------------------------------------------------------------------
# GroupTokenBackbone (top-level)
# ---------------------------------------------------------------------------

class GroupTokenBackbone(nn.Module):
    """GroupToken-HilbertAttention backbone.

    Replaces MLGC feat_extract + unit_coupling. Outputs aug (B,N,48)
    and inj_f List[12×(B,N,51)].
    """

    def __init__(
        self,
        pc_channel: int = 3,
        aug_channel: int = 48,
        n_injector: int = 12,
        inj_channel: int = 51,
        G: int = 64,
        S: int = 32,
        C: int = 128,
        depth: int = 4,
        heads: int = 4,
        upsample_k: int = 8,
        point_identity_dim: int = 0,
        point_identity_gamma_init: float = 0.05,
        upsample_knn_backend: str = "numpy",
        build_heads: bool = True,
        local_norm: str = "batch",
    ):
        super().__init__()
        self.G = G
        self.S = S
        self.C = C
        self.depth = depth
        self.n_injector = n_injector
        self.aug_channel = aug_channel
        self.inj_channel = inj_channel
        self.point_identity_dim = point_identity_dim
        self.point_identity_gamma_init = float(point_identity_gamma_init)

        # 1. Local group encoder
        self.local_encoder = _LocalEncoder(out_dim=C, norm=local_norm)

        # 2. Center position embedding
        self.center_pos_mlp = nn.Sequential(
            nn.Linear(3, C // 2),
            nn.ReLU(),
            nn.Linear(C // 2, C),
        )

        # 3. Token mixer
        self.mixer = _TokenMixer(dim=C, depth=depth, heads=heads)

        # 4. Cross-attention upsample
        self.upsample = _CrossAttentionUpsample(dim=C, k=upsample_k, knn_backend=upsample_knn_backend)

        # 5. Point identity bypass (preserves per-point info through bottleneck)
        if point_identity_dim > 0:
            self.point_id_mlp = _PointIdentityMLP(
                in_dim=3, hidden=32, out_dim=point_identity_dim
            )
            self.point_id_gamma = jt.array([self.point_identity_gamma_init])
            self.id_fusion = nn.Sequential(
                nn.Linear(C + point_identity_dim, C),
                nn.ReLU(),
            )
        else:
            self.point_id_mlp = None
            self.point_id_gamma = None
            self.id_fusion = None

        # 6. Output heads（displacement head displacement 模式 build_heads=False 时不建，省去无梯度 INN 头）
        if build_heads:
            self.aug_head = nn.Sequential(
                nn.Linear(C, C),
                nn.ReLU(),
                nn.Linear(C, aug_channel),
            )
            self.inj_heads = nn.ModuleList()
            for _ in range(n_injector):
                self.inj_heads.append(nn.Sequential(
                    nn.Linear(C, C),
                    nn.ReLU(),
                    nn.Linear(C, inj_channel),
                ))
        else:
            self.aug_head = None
            self.inj_heads = None

        # Diagnostics
        self._last_diag: Dict[str, float] = {}

    def execute(self, xyz: jt.Var, return_point_feat: bool = False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) noisy patch coordinates (centered)
            return_point_feat: if True, return trunk point features (B, N, C) and
                skip aug/inj heads + diagnostics. Used by displacement head displacement head;
                default False preserves 1.13 (inj_f, aug) output.

        Returns:
            return_point_feat=False: (inj_f: List[12×(B,N,inj)], aug: (B,N,aug))
            return_point_feat=True:  point_feat (B, N, C)
        """
        B, N, _ = xyz.shape

        # ---- 1. FPS + KNN grouping ----
        centers, center_idx = _sample_centers(xyz, self.G)  # (B,G,3), (B,G)
        _, knn_idx = safe_knn(centers, xyz, self.S)   # (B, G, S)

        # Gather neighborhood, center by subtract
        bi = jt.arange(B).unsqueeze(-1).unsqueeze(-1).broadcast([B, self.G, self.S])
        groups = xyz[bi, knn_idx]                         # (B, G, S, 3)
        groups = groups - centers.unsqueeze(2)            # center

        # ---- 2. Local encoder ----
        tokens = self.local_encoder(groups)               # (B, G, C)

        # ---- 3. Position embedding ----
        token_pos = self.center_pos_mlp(centers)          # (B, G, C)

        # ---- 4. Hilbert serialization (bidirectional) ----
        centers_np = np.asarray(centers.stop_grad().numpy())
        # Forward: standard Hilbert on (x, y, z)
        sort_fw, _ = hilbert_sort_indices(centers_np, bits=10)
        # Backward: hilbert-trans Hilbert on (y, x, z) — swapped x/y axes
        centers_swapped = centers_np[:, :, [1, 0, 2]]
        sort_bw, _ = hilbert_sort_indices(centers_swapped, bits=10)

        # Gather indices
        bi_g = jt.arange(B).unsqueeze(-1).broadcast([B, self.G])  # (B, G)

        # Forward direction
        tokens_fw = tokens[bi_g, sort_fw]       # (B, G, C)
        pos_fw = token_pos[bi_g, sort_fw]       # (B, G, C)
        centers_fw = centers[bi_g, sort_fw]     # (B, G, 3)
        # Backward direction (hilbert-trans order)
        tokens_bw = tokens[bi_g, sort_bw]       # (B, G, C)
        pos_bw = token_pos[bi_g, sort_bw]       # (B, G, C)
        centers_bw = centers[bi_g, sort_bw]     # (B, G, 3)

        # Concat forward + backward → (B, 2G, C)
        mixer_tokens = jt.concat([tokens_fw, tokens_bw], dim=1)
        mixer_pos = jt.concat([pos_fw, pos_bw], dim=1)

        # ---- 5. Token mixer ----
        x_mix = mixer_tokens + mixer_pos
        x_mix, mixer_entropies = self.mixer(x_mix)        # (B, 2G, C)

        # ---- 6. Cross-attention upsample ----
        # token_centers must follow the same order as x_mix.
        token_centers = jt.concat([centers_fw, centers_bw], dim=1)  # (B, 2G, 3)
        point_feat, upsample_entropy, topk_idx = self.upsample(
            x_mix, token_centers, xyz
        )  # (B, N, C)

        # ---- 6.5. Point identity bypass ----
        if self.point_id_mlp is not None:
            point_id = self.point_id_mlp(xyz)              # (B, N, point_identity_dim)
            point_id_norm = jt.sqrt((point_id ** 2).sum(dim=-1)).mean()
            point_feat_norm = jt.sqrt((point_feat ** 2).sum(dim=-1)).mean()
            point_id_gamma = self.point_id_gamma.clamp(0.0, 0.25)
            point_id_scaled = point_id * point_id_gamma
            point_feat = self.id_fusion(
                jt.concat([point_feat, point_id_scaled], dim=-1)
            )  # (B, N, C)
            id_contrib = float(
                ((point_id_gamma.abs() * point_id_norm) / (point_feat_norm + point_id_gamma.abs() * point_id_norm + 1e-8)).item()
            )
            id_gamma = float(point_id_gamma.item())
        else:
            point_id_norm = jt.zeros((1,))
            point_feat_norm = jt.ones((1,))
            id_contrib = 0.0
            id_gamma = 0.0

        # ---- 6.6. displacement head displacement-head hook ----
        if return_point_feat:
            return point_feat

        # ---- 7. Heads ----
        aug = self.aug_head(point_feat)                   # (B, N, 48)
        inj_f = [head(point_feat) for head in self.inj_heads]  # 12 × (B, N, 51)

        # ---- Diagnostics ----
        delta_norm = jt.sqrt((aug ** 2).sum(dim=-1)).mean()
        top1_np = np.asarray(topk_idx[:, :, 0].stop_grad().numpy()).astype(np.int64)
        token_usage = []
        center_usage = []
        for b in range(B):
            token_center_id = np.concatenate([sort_fw[b], sort_bw[b]], axis=0).astype(np.int64)
            token_usage.append(len(np.unique(top1_np[b])) / float(2 * self.G))
            center_usage.append(len(np.unique(token_center_id[top1_np[b]])) / float(self.G))

        self._last_diag = {
            "token_C": float(self.C),
            "token_G": float(self.G),
            "token_2G": float(2 * self.G),
            "aug_delta_norm": float(delta_norm.item()),
            "mixer_entropy_mean": float(np.mean(mixer_entropies)) if mixer_entropies else 0.0,
            "upsample_entropy_mean": upsample_entropy,
            "upsample_token_usage_ratio": float(np.mean(token_usage)),
            "upsample_center_usage_ratio": float(np.mean(center_usage)),
            "point_id_dim": float(self.point_identity_dim),
            "point_id_gamma": float(id_gamma),
            "point_id_norm": float(point_id_norm.item()),
            "point_feat_fusion_norm": float(point_feat_norm.item()),
            "point_id_contrib_ratio": float(id_contrib),
        }

        return inj_f, aug

    def last_diag(self) -> Dict[str, float]:
        return dict(self._last_diag)
