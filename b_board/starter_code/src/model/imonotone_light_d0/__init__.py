"""与原版 PD-LTS Light 比较的 iMonotone 组件，未用于最终提交。

使用 coeff=0.98、LipSwish（可学习 beta 的激活）和 RootFind（不动点求解），
用于前向数值对照。复用 imonotone_light 的 ActNorm、InducedNormLinear
和 find_fixed_point，另定义 FCNet、单调块及 FlowAssembly。

monotone_block.py 用于前向比较，不提供原版的隐式反向传播；
monotone_block_trainable.py 使用暖启动后展开 Banach 迭代的近似梯度。
"""

from .activations import LipSwish
from .fcnet import FCNetD0
from .monotone_block import iMonotoneBlockD0
from .flow_assembly import IMonotoneFlowAssemblyD0

__all__ = [
    "LipSwish",
    "FCNetD0",
    "iMonotoneBlockD0",
    "IMonotoneFlowAssemblyD0",
]
