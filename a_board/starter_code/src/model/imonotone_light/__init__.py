"""iMonotone（单调可逆层）的 Light 结构对照组件。

来源：PD-LTS 的 models/model_light/deflow.py::FlowAssembly。
结构为 [iMonotoneBlock, ActNorm, iMonotoneBlock(preact), ActNorm]。
与 pdlts_heavy 复用早期移植组件，但不依赖其训练系统或数据处理接口。
本目录 ActNorm 在初始化时分离输入统计量，保留参数梯度。

最终模型使用仿射耦合层；单调块的结构和损失对照见
experiments/overview.md。
"""

from .flow_assembly import IMonotoneFlowAssembly

__all__ = ["IMonotoneFlowAssembly"]
