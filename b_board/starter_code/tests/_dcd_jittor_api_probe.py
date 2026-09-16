"""Jittor API probe for validating the dcd_official implementation.

Validation questions:
  - jt.misc.knn returns squared distance or euclidean? Must NOT guess.
  - jt.scatter_add_ / equivalent for batched bincount.
"""

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0  # CPU is enough for API probing
jt.set_global_seed(42)

# ============================================================
# (1) jt.misc.knn distance type
# ============================================================
np.random.seed(0)
src_np = np.array([[[0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0]]], dtype=np.float32)  # (1, 2, 3)
tgt_np = np.array([[[1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0]]], dtype=np.float32)  # (1, 3, 3)
src = jt.array(src_np)
tgt = jt.array(tgt_np)

dist, idx = jt.misc.knn(src, tgt, 1)
print("jt.misc.knn signature returns: (dist, idx)")
print(f"  dist shape={tuple(dist.shape)}  dtype={dist.dtype}")
print(f"  idx  shape={tuple(idx.shape)}   dtype={idx.dtype}")
print(f"  dist values: {dist.numpy()}")
print(f"  idx  values: {idx.numpy()}")

# Expected NN per src point:
# src[0] = (0,0,0). Distances: to (1,0,0) = 1 (sq=1, eucl=1);
#                              to (0,0,0) = 0; to (10,10,10) = 300 (sq) / sqrt(300) (eucl).
#   nearest = tgt[1], dist_sq=0, dist_eucl=0
# src[1] = (3,4,0). Distances: to (1,0,0) = 4+16+0 = 20 (sq), sqrt(20)~4.47 (eucl);
#                              to (0,0,0) = 9+16+0 = 25 (sq), 5 (eucl);
#                              to (10,10,10) = 49+36+100 = 185 (sq).
#   nearest = tgt[0], dist_sq=20, dist_eucl=sqrt(20)~4.4721
print("\nReference: src[0]→tgt[1]: sq=0, eucl=0")
print("           src[1]→tgt[0]: sq=20, eucl=4.4721359...")
print()

# Diagnose: compare dist[0,1] to 20 vs sqrt(20)
d_observed = float(dist.numpy()[0, 1, 0]) if dist.numpy().ndim == 3 else float(dist.numpy()[0, 1])
print(f"Observed dist[0,1] = {d_observed}")
if abs(d_observed - 20.0) < 1e-3:
    print(">>> CONCLUSION: jt.misc.knn returns SQUARED distance (matches official chamfer3D semantics)")
elif abs(d_observed - np.sqrt(20.0)) < 1e-3:
    print(">>> CONCLUSION: jt.misc.knn returns EUCLIDEAN distance (we must square it ourselves)")
else:
    print(f">>> UNKNOWN: neither sq=20 nor eucl=4.472 matches; got {d_observed}")

# ============================================================
# (2) Jittor batched scatter_add equivalent
# ============================================================
print("\n" + "=" * 60)
print("Probing Jittor scatter_add API")
print("=" * 60)

# Goal: implement count[b, k] += 1 for each (b, idx[b, n]).
# In NumPy: np.add.at(count, (batch_idx, idx), 1)
# In PyTorch (utils_v2): count.scatter_add_(1, idx.long(), torch.ones_like(idx))

# Method A: try jt.Var.scatter_
B, N, M = 2, 5, 4
np.random.seed(1)
idx_np = np.random.randint(0, M, size=(B, N)).astype(np.int64)
print(f"idx (B={B}, N={N}): {idx_np}")

idx = jt.array(idx_np)
count = jt.zeros((B, M), dtype=jt.int64)

# Try the scatter_add_ pattern (PyTorch-style)
try:
    ones = jt.ones((B, N), dtype=jt.int64)
    count_a = count.scatter_(1, idx, ones, reduce="add")
    print(f"scatter_(reduce='add') returned shape {tuple(count_a.shape)}: {count_a.numpy()}")
except Exception as e:
    print(f"scatter_(reduce='add') failed: {type(e).__name__}: {e}")

# Method B: use jt.scatter / jt.misc.scatter
try:
    count_b = jt.zeros((B, M), dtype=jt.int64)
    ones = jt.ones((B, N), dtype=jt.int64)
    res = count_b.scatter_add(1, idx, ones)  # not in-place
    print(f"scatter_add returned shape {tuple(res.shape)}: {res.numpy()}")
except Exception as e:
    print(f"scatter_add failed: {type(e).__name__}: {e}")

# Method C: per-batch python loop with jt.bincount-equivalent
# Fallback: iterate over batch in Python (slow but correct).
print("\nReference (NumPy np.add.at):")
ref_count = np.zeros((B, M), dtype=np.int64)
for b in range(B):
    for n in range(N):
        ref_count[b, idx_np[b, n]] += 1
print(ref_count)
