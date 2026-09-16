"""PDLTS Light network (Jittor 移植)

算法主体依据上游 PyTorch 参考实现移植到 Jittor。
结构和损失对照见 experiments/overview.md。
注册到：starter_code/src/model/parse.py 的 MAP（模型注册入口）。
"""

from .layer import (
    knn_group,
    get_knn_idx,
    FullyConnectedLayer,
    noiseEdgeConv,
    PreConv,
    EdgeConv,
    FeatMergeUnit,
)
from .inn import (
    ActNorm1d,
    InvertibleLinear,
    AffineCoupling,
    SequentialFlow,
    FlowAssembly,
)
from .hilbert import hilbert_encode, hilbert_sort_indices
from .global_context import GlobalContextBlock
from .group_token_backbone import GroupTokenBackbone
from .model import PDLTSLightNetwork
from .system import PDLTSLight

__all__ = [
    # MLGC (layer.py)
    "knn_group",
    "get_knn_idx",
    "FullyConnectedLayer",
    "noiseEdgeConv",
    "PreConv",
    "EdgeConv",
    "FeatMergeUnit",
    # INN (inn.py)
    "ActNorm1d",
    "InvertibleLinear",
    "AffineCoupling",
    "SequentialFlow",
    "FlowAssembly",
    # Hilbert encoding (hilbert.py)
    "hilbert_encode",
    "hilbert_sort_indices",
    # Global context (global_context.py)
    "GlobalContextBlock",
    # GroupToken backbone (group_token_backbone.py)
    "GroupTokenBackbone",
    # 整网 (model.py)
    "PDLTSLightNetwork",
    # ModelSpec 包装 (system.py, 注册到 src/model/parse.py MAP)
    "PDLTSLight",
]
