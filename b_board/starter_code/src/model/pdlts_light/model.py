"""PDLTS Light 网络：局部图特征、可逆层与潜空间噪声分离。

基于 PD-LTS 的 models/model_light/deflow.py::DenoiseFlow 移植。
可逆层使用 ActNorm1d（激活归一化）和 AffineCoupling（仿射耦合），
潜空间使用 FBM（固定二值掩码）；未实现 LBM/LCC 分支。
FBM 分支的 loss_denoise 恒为 0，与原版该分支一致。
结构和损失消融见 experiments/overview.md。

输入 patches 的形状为 (B, N, 3)，返回 denoised、ldj、loss_denoise。
数据处理和损失计算由 ModelSpec 负责，训练循环由 PDLTSLightSystem 负责。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import jittor as jt
from jittor import nn

from .layer import (
    EdgeConv,
    FeatMergeUnit,
    PreConv,
    noiseEdgeConv,
    safe_knn,
)
from .inn import FlowAssembly
from .global_context import GlobalContextBlock
from .group_token_backbone import GroupTokenBackbone


        # iMonotone baseline: config-driven FlowAssembly 族选择
# 默认 affine_coupling = 现有行为（ActNorm1d+AffineCoupling），零变化；
# imonotone = IMonotoneFlowAssembly（iMonotoneBlock-based），历史 iMonotone 对照测试用。
_FLOW_ASSEMBLY_KINDS = ("affine_coupling", "imonotone")


# Light 真实规格 (来自 train_deflow_score.py argparse 默认值，非 deflow.py 类签名默认值)
LIGHT_AUG_CHANNEL = 48
LIGHT_N_INJECTOR = 12
LIGHT_CUT_CHANNEL = 24
LIGHT_NFLOW_MODULE = 12
LIGHT_NUM_NEIGHBORS = 32

# MLGC 注入路径的通道表 (deflow.py:113-118)
# 每张表长度必须等于 n_injector = 12
_IN_CHANNEL_E = [16, 48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168]
_IN_CHANNEL_A = [48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168, 96]
_OUT_CHANNEL = [32, 32, 32, 32, 32, 32, 32, 96, 24, 24, 24, 96]
# concat=False 的层编号 (deflow.py:118)
_CONCAT_FALSE_INDEX = {7, 11}
_MLGC_HIDDEN = 64


class PDLTSLightNetwork(nn.Module):
    """PDLTS Light 整网。输入 noisy patch (B, N, 3)，输出 denoised patch (B, N, 3)。

    流程对应 DenoiseFlow.forward (deflow.py:206-235):
        1. feat_extract(xyz)             -> 12 个 injection feature inj_f[i], 每个 (B, N, 51)
        2. unit_coupling(xyz)            -> aug_feat (B, N, 48)
        3. x = cat(xyz, aug_feat)        -> (B, N, 51)
        4. z = f(x, inj_f)               -> 12 层 FlowAssembly 前向，前 12 层加 inj_f[i]
        5. FBM: z[..., -cut_channel:] = 0  (最后 24 通道置零)
        6. denoised = sample(z, inj_f) = g(z, inj_f)[..., :3]
    """

    def __init__(
        self,
        pc_channel: int = 3,
        aug_channel: int = LIGHT_AUG_CHANNEL,
        n_injector: int = LIGHT_N_INJECTOR,
        cut_channel: int = LIGHT_CUT_CHANNEL,
        fbm_mask_mode: str = "hard",
        fbm_soft_tail_init: float = 0.05,
        fbm_topk_init_margin: float = 0.5,
        nflow_module: int = LIGHT_NFLOW_MODULE,
        num_neighbors: int = LIGHT_NUM_NEIGHBORS,
        mlgc_hidden: int = _MLGC_HIDDEN,
        mlgc_norm_mode: str = "batch",
        mlgc_group_norm_max_groups: int = 8,
        coupling_hidden: int = 64,
        log_scale_clamp: float = 0.1,
        direction_head_mode: str = "off",
        direction_head_hidden: int = 64,
        direction_head_feat_source: str = "predict_z",
        # 全局上下文模块
        global_context_mode: str = "off",
        global_context_d_model: int = 64,
        global_context_n_heads: int = 4,
        global_context_n_layers: int = 2,
        global_context_ffn_multiplier: int = 2,
        # GroupToken 骨干网络配置
        backbone_mode: str = "mlgc",
        backbone_G: int = 64,
        backbone_S: int = 32,
        backbone_C: int = 128,
        backbone_depth: int = 4,
        backbone_heads: int = 4,
        backbone_upsample_k: int = 8,
        backbone_point_id_dim: int = 0,
        backbone_point_id_gamma_init: float = 0.05,
        # post-FBM residual head
        residual_head_mode: str = "off",
        residual_head_hidden: int = 64,
        # soft keep-weight V1: soft keep-weight head
        keep_head_mode: str = "off",
        keep_head_hidden: int = 64,
        # 候选生成、选择与精修（INN 之后、patch 级）
        # 默认全 off → 保持基础模型输出不变。三段开关独立：
        #   candidate_mode: off | slot_variant （每点生成 R 个变体，不改点数）
        #   selector_mode:  off | identity | learned（逐点在 R 变体内选 1；identity=选 base）
        #   cleaner_mode:   off | residual     （选后残差精修，zero-init）
        candidate_mode: str = "off",
        candidate_R: int = 4,
        candidate_knn_k: int = 6,
        selector_mode: str = "off",
        selector_hidden: int = 64,
        cleaner_mode: str = "off",
        cleaner_hidden: int = 64,
        # confidence-gated pull head (default-off).
        pull_head_mode: str = "off",
        pull_head_hidden: int = 64,
        pull_delta_max: float = 0.0,
        pull_head_feat_source: str = "predict_z",
        # iMonotone baseline: FlowAssembly 族选择（默认 affine_coupling = 零变化）
        flow_assembly_kind: str = "affine_coupling",
    ):
        super().__init__()

        # 确认通道表与 n_injector 匹配
        assert n_injector == len(_IN_CHANNEL_E) == len(_IN_CHANNEL_A) == len(_OUT_CHANNEL), (
            f"channel tables length != n_injector={n_injector}, "
            f"got E={len(_IN_CHANNEL_E)}, A={len(_IN_CHANNEL_A)}, O={len(_OUT_CHANNEL)}"
        )
        assert nflow_module >= n_injector, (
            f"nflow_module ({nflow_module}) must be >= n_injector ({n_injector}), "
            f"否则前 n_injector 层的注入没有对应的 FlowAssembly"
        )
        assert cut_channel < pc_channel + aug_channel, (
            f"cut_channel ({cut_channel}) must be < pc+aug ({pc_channel + aug_channel})"
        )

        self.pc_channel = pc_channel
        self.aug_channel = aug_channel
        self.n_injector = n_injector
        self.cut_channel = cut_channel
        self.fbm_mask_mode = str(fbm_mask_mode).lower()
        self.fbm_soft_tail_init = float(fbm_soft_tail_init)
        self.fbm_topk_init_margin = float(fbm_topk_init_margin)
        self.nflow_module = nflow_module
        self.num_neighbors = num_neighbors

        assert self.fbm_mask_mode in (
            "hard", "learnable_soft_tail", "learnable_topk_ste"
        ), (
            "fbm_mask_mode must be hard/learnable_soft_tail/learnable_topk_ste, got "
            f"{self.fbm_mask_mode!r}"
        )
        if self.fbm_mask_mode == "learnable_soft_tail":
            assert cut_channel > 0, "learnable_soft_tail requires cut_channel > 0"
            assert 0.0 < self.fbm_soft_tail_init < 1.0, (
                "fbm_soft_tail_init must be in (0, 1), got "
                f"{self.fbm_soft_tail_init}"
            )
        if self.fbm_mask_mode == "learnable_topk_ste":
            assert cut_channel > 0, "learnable_topk_ste requires cut_channel > 0"
            assert self.fbm_topk_init_margin > 0.0, (
                "fbm_topk_init_margin must be positive, got "
                f"{self.fbm_topk_init_margin}"
            )

        inj_channel = pc_channel + aug_channel  # = 51 for light

        # ----- backbone family -----
        self.backbone_mode = str(backbone_mode)
        assert self.backbone_mode in ("mlgc", "group_token_hilbert"), (
            f"backbone_mode must be mlgc/group_token_hilbert, got {self.backbone_mode!r}"
        )

        if self.backbone_mode == "mlgc":
            mlgc_norm_mode = str(mlgc_norm_mode).lower()
            assert mlgc_norm_mode in ("batch", "group", "layer", "rms"), (
                "mlgc_norm_mode must be batch/group/layer/rms, got "
                f"{mlgc_norm_mode!r}"
            )
            assert int(mlgc_group_norm_max_groups) >= 1, (
                "mlgc_group_norm_max_groups must be positive, got "
                f"{mlgc_group_norm_max_groups}"
            )
            # ----- MLGC 特征提取 (旧路径, 完全不动) -----
            # unit_coupling: xyz -> aug feature
            self.noise_params = noiseEdgeConv(
                in_channel=pc_channel, hidden_channel=32, out_channel=aug_channel
            )
            # PreConv: xyz -> 16 通道入口
            self.PreConv = PreConv(
                in_channel=pc_channel,
                out_channel=16,
                norm_mode=mlgc_norm_mode,
                group_norm_max_groups=mlgc_group_norm_max_groups,
            )
            # n_injector 层 EdgeConv + FeatMergeUnit
            self.feat_Conv = nn.ModuleList()
            self.AdaptConv = nn.ModuleList()
            for i in range(n_injector):
                concat = i not in _CONCAT_FALSE_INDEX
                self.feat_Conv.append(EdgeConv(
                    in_channel=_IN_CHANNEL_E[i],
                    hidden_channel=mlgc_hidden,
                    out_channel=_OUT_CHANNEL[i],
                    concat=concat,
                    norm_mode=mlgc_norm_mode,
                    group_norm_max_groups=mlgc_group_norm_max_groups,
                ))
                self.AdaptConv.append(FeatMergeUnit(
                    in_channel=_IN_CHANNEL_A[i],
                    hidden_channel=mlgc_hidden,
                    out_channel=inj_channel,
                    norm_mode=mlgc_norm_mode,
                    group_norm_max_groups=mlgc_group_norm_max_groups,
                ))
            self.group_token_backbone = None

        elif self.backbone_mode == "group_token_hilbert":
            # ----- GroupToken-HilbertAttention backbone -----
            self.group_token_backbone = GroupTokenBackbone(
                pc_channel=pc_channel,
                aug_channel=aug_channel,
                n_injector=n_injector,
                inj_channel=inj_channel,
                G=int(backbone_G),
                S=int(backbone_S),
                C=int(backbone_C),
                depth=int(backbone_depth),
                heads=int(backbone_heads),
                upsample_k=int(backbone_upsample_k),
                point_identity_dim=int(backbone_point_id_dim),
                point_identity_gamma_init=float(backbone_point_id_gamma_init),
            )
            # MLGC 模块不创建, 设为 None 防止误用
            self.noise_params = None
            self.PreConv = None
            self.feat_Conv = nn.ModuleList()
            self.AdaptConv = nn.ModuleList()

        # ----- INN 主干 -----
        self.flow_assembly_kind = str(flow_assembly_kind)
        assert self.flow_assembly_kind in _FLOW_ASSEMBLY_KINDS, (
            f"flow_assembly_kind must be one of {_FLOW_ASSEMBLY_KINDS}, "
            f"got {self.flow_assembly_kind!r}"
        )
        if self.flow_assembly_kind == "affine_coupling":
            self.flow_assemblies = nn.ModuleList([
                FlowAssembly(
                    channel=inj_channel,
                    hidden=coupling_hidden,
                    log_scale_clamp=log_scale_clamp,
                )
                for _ in range(nflow_module)
            ])
        elif self.flow_assembly_kind == "imonotone":
            from ..imonotone_light import IMonotoneFlowAssembly
            self.flow_assemblies = nn.ModuleList([
                IMonotoneFlowAssembly(channel=inj_channel)
                for _ in range(nflow_module)
            ])

        # ----- FBM disentangle -----
        # 原仓库用 nn.Parameter(requires_grad=False) 存 channel_mask。
        # 当前实现直接写死 "最后 cut_channel 个通道 = 0"。不暴露为参数。
        # 保留 mask 字段，便于后续扩展到 LBM 时替换。
        mask = np.ones((1, 1, inj_channel), dtype=np.float32)
        mask[..., -cut_channel:] = 0.0
        self.channel_mask = jt.array(mask).stop_grad()
        self.fbm_tail_logit = None
        self.fbm_mask_logit = None
        if self.fbm_mask_mode == "learnable_soft_tail":
            init_logit = np.log(
                self.fbm_soft_tail_init / (1.0 - self.fbm_soft_tail_init)
            )
            self.fbm_tail_logit = jt.array(np.full(
                (1, 1, cut_channel), init_logit, dtype=np.float32
            ))
        elif self.fbm_mask_mode == "learnable_topk_ste":
            # 初始硬前向严格复现原 FBM；训练只允许交换身份，保留数固定为 C-cut。
            init_logits = np.full(
                (1, 1, inj_channel),
                -self.fbm_topk_init_margin,
                dtype=np.float32,
            )
            init_logits[..., :-cut_channel] = self.fbm_topk_init_margin
            self.fbm_mask_logit = jt.array(init_logits)

        # ----- direction-magnitude head (direction diagnostics) -----
        self.direction_head_mode = str(direction_head_mode)
        self.direction_head_hidden = int(direction_head_hidden)
        self.direction_head_feat_source = str(direction_head_feat_source)
        assert self.direction_head_mode in ("off", "replace_output", "auxiliary"), (
            f"direction_head_mode must be off/replace_output/auxiliary, got {self.direction_head_mode!r}"
        )
        assert self.direction_head_feat_source in ("predict_z", "z", "predict_z_inj_last"), (
            f"direction_head_feat_source must be predict_z/z/predict_z_inj_last, got {self.direction_head_feat_source!r}"
        )
        if self.direction_head_mode in ("replace_output", "auxiliary"):
            if self.direction_head_feat_source == "predict_z_inj_last":
                head_input_dim = 2 * inj_channel
            else:
                head_input_dim = inj_channel
            self.head_input_dim = head_input_dim
            self.MLP_dir = nn.Sequential(
                nn.Linear(head_input_dim, direction_head_hidden),
                nn.ReLU(),
                nn.Linear(direction_head_hidden, pc_channel),
            )
            self.MLP_mag = nn.Sequential(
                nn.Linear(head_input_dim, direction_head_hidden),
                nn.ReLU(),
                nn.Linear(direction_head_hidden, 1),
            )
        # direction diagnostic stash (log-only, set by execute)
        self._dh1_denoised_implicit: Optional[jt.Var] = None
        self._dh1_head_unit_dir: Optional[jt.Var] = None
        self._dh1_head_mag: Optional[jt.Var] = None

        # ----- post-FBM residual head -----
        self.residual_head_mode = str(residual_head_mode)
        assert self.residual_head_mode in ("off", "post_fbm"), (
            f"residual_head_mode must be off/post_fbm, got {self.residual_head_mode!r}"
        )
        if self.residual_head_mode == "post_fbm":
            # input: concat(denoised_implicit(3), predict_z(51)) = 54 dims
            head_input_dim = pc_channel + pc_channel + aug_channel  # 3 + 3 + 48 = 54
            h = int(residual_head_hidden)
            self.residual_mlp = nn.Sequential(
                nn.Linear(head_input_dim, h),
                nn.ReLU(),
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Linear(h // 2, pc_channel),
            )
            # zero-init final layer
            last_linear = self.residual_mlp[-1]  # nn.Sequential 按 __getitem__ 取
            jt.init.constant_(last_linear.weight, 0.0)
            jt.init.constant_(last_linear.bias, 0.0)
        # diagnostic stash
        self._residual_delta_norm: Optional[float] = None
        self._residual_delta_ratio: Optional[float] = None

        # ----- soft keep-weight V1: per-point keep-weight head -----
        self.keep_head_mode = str(keep_head_mode)
        assert self.keep_head_mode in ("off", "soft_weight"), (
            f"keep_head_mode must be off/soft_weight, got {self.keep_head_mode!r}"
        )
        self.MLP_keep = None
        if self.keep_head_mode == "soft_weight":
            h = int(keep_head_hidden)
            self.MLP_keep = nn.Sequential(
                nn.Linear(inj_channel, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
            last_linear = self.MLP_keep[-1]
            jt.init.constant_(last_linear.weight, 0.0)
            jt.init.constant_(last_linear.bias, 0.0)
        self._keep_logit: Optional[jt.Var] = None

        # ----- 候选生成、选择与精修 -----
        # slot_variant：每点 R 变体，selector 逐点选 1，输出第 i 槽恒为点 i 的精修，保槽位。
        self.candidate_mode = str(candidate_mode)
        self.candidate_R = int(candidate_R)
        self.candidate_knn_k = int(candidate_knn_k)
        self.selector_mode = str(selector_mode)
        self.cleaner_mode = str(cleaner_mode)
        assert self.candidate_mode in ("off", "slot_variant"), (
            f"candidate_mode must be off/slot_variant, got {self.candidate_mode!r}"
        )
        assert self.selector_mode in ("off", "identity", "learned"), (
            f"selector_mode must be off/identity/learned, got {self.selector_mode!r}"
        )
        assert self.cleaner_mode in ("off", "residual"), (
            f"cleaner_mode must be off/residual, got {self.cleaner_mode!r}"
        )
        # selector/cleaner 需要变体；候选 off 时它们必须也 off
        if self.selector_mode != "off" or self.cleaner_mode != "off":
            assert self.candidate_mode != "off", (
                "selector_mode/cleaner_mode 需要 candidate_mode != off 提供变体"
            )
        # learned selector：per-variant 打分 MLP（输入 = 变体坐标 3 + 该点 predict_z）。
        self.MLP_selector = None
        if self.selector_mode == "learned":
            h = int(selector_hidden)
            self.MLP_selector = nn.Sequential(
                nn.Linear(pc_channel + inj_channel, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
            # zero-init 末层 → 初始所有变体同分，argmax 取 variant 0(=base)，初始等价 base
            jt.init.constant_(self.MLP_selector[-1].weight, 0.0)
            jt.init.constant_(self.MLP_selector[-1].bias, 0.0)
        # cleaner：选后 per-point 残差精修 MLP（zero-init → 初始 identity）
        self.MLP_cleaner = None
        if self.cleaner_mode == "residual":
            h = int(cleaner_hidden)
            self.MLP_cleaner = nn.Sequential(
                nn.Linear(pc_channel + inj_channel, h),
                nn.ReLU(),
                nn.Linear(h, pc_channel),
            )
            jt.init.constant_(self.MLP_cleaner[-1].weight, 0.0)
            jt.init.constant_(self.MLP_cleaner[-1].bias, 0.0)

        # ----- confidence-gated bounded pull head -----
        # This is deliberately constructed after the trunk and all legacy heads:
        # B(off) does not create parameters, while C(on) cannot perturb trunk RNG.
        self.pull_head_mode = str(pull_head_mode)
        self.pull_head_hidden = int(pull_head_hidden)
        self.pull_delta_max = float(pull_delta_max)
        self.pull_head_feat_source = str(pull_head_feat_source)
        assert self.pull_head_mode in ("off", "confidence_delta"), (
            f"pull_head_mode must be off/confidence_delta, got {self.pull_head_mode!r}"
        )
        assert self.pull_head_feat_source in (
            "predict_z",
            "z",
            "denoised_implicit",
            "base",
            "xyz",
            "predict_z_base",
        ), (
            "pull_head_feat_source must be predict_z/z/denoised_implicit/"
            f"base/xyz/predict_z_base, got {self.pull_head_feat_source!r}"
        )
        self.MLP_pull = None
        if self.pull_head_mode == "confidence_delta":
            if self.pull_head_feat_source in ("predict_z", "z"):
                pull_input_dim = inj_channel
            elif self.pull_head_feat_source in ("denoised_implicit", "base", "xyz"):
                pull_input_dim = pc_channel
            else:
                pull_input_dim = inj_channel + pc_channel
            h = int(pull_head_hidden)
            self.MLP_pull = nn.Sequential(
                nn.Linear(pull_input_dim, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, pc_channel + 1),
            )
            # Initial C path is identity: pull=0, w=0.5, denoised unchanged.
            jt.init.constant_(self.MLP_pull[-1].weight, 0.0)
            jt.init.constant_(self.MLP_pull[-1].bias, 0.0)
        self._pull_w: Optional[jt.Var] = None
        self._pull_norm: Optional[jt.Var] = None
        self._pull_vector: Optional[jt.Var] = None
        self._pull_input_audit: Dict[str, object] = {
            "pull_head_mode": self.pull_head_mode,
            "pull_head_feat_source": self.pull_head_feat_source,
            "allowed_inputs_only": True,
            "input_tags": [],
        }
        # diagnostic / loss stash（execute 设置；off 时为 None）
        self._variant_logit: Optional[jt.Var] = None        # (B, M, R) selector 逐点变体分数
        self._selected_variant_idx: Optional[jt.Var] = None  # (B, M) 每点选了第几个变体
        self._slot_variants: Optional[jt.Var] = None         # (B, M, R, 3) 变体张量（供 loss）

        # ----- 全局上下文模块 -----
        self.global_context_mode = str(global_context_mode)
        assert self.global_context_mode in ("off", "hilbert_full"), (
            f"global_context_mode must be off/hilbert_full, got {self.global_context_mode!r}"
        )
        assert not (
            self.backbone_mode == "group_token_hilbert" and self.global_context_mode != "off"
        ), "group_token_hilbert already replaces MLGC; keep global_context_mode='off' for the single-variable GroupToken backbone."
        if self.global_context_mode != "off":
            self.global_block = GlobalContextBlock(
                feature_dim=aug_channel,
                d_model=int(global_context_d_model),
                n_heads=int(global_context_n_heads),
                n_layers=int(global_context_n_layers),
                ffn_multiplier=int(global_context_ffn_multiplier),
                max_patch_size=2048,  # safe upper bound for any patch size
            )
        else:
            self.global_block = None
        self._global_diag: Dict[str, float] = {}

    # ----- MLGC path -----
    def unit_coupling(self, xyz: jt.Var, knn_idx: jt.Var) -> jt.Var:
        """xyz -> aug feature (B, N, aug_channel)。对应 deflow.py:237-241。"""
        return self.noise_params(xyz, knn_idx)

    def feat_extract(self, xyz: jt.Var, knn_idx: jt.Var) -> List[jt.Var]:
        """xyz -> n_injector 个 injection feature, 每个 (B, N, inj_channel)。

        对应 deflow.py:243-251。
        """
        cs: List[jt.Var] = []
        f = self.PreConv(xyz, knn_idx)
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, knn_idx)
            inj_f = self.AdaptConv[i](f)
            cs.append(inj_f)
        return cs

    # ----- INN path -----
    def f(self, x: jt.Var, inj_f: List[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        """INN forward with injection。

        对应 deflow.py:171-181。当前基础方案不计算精确 log_det_J，返回 0 张量占位。
        (真实 ldj 已经在各 FlowAssembly 内部正确计算 — 若需要，用 logpx 重载接口。)
        """
        B = x.shape[0]
        log_det_J = jt.zeros((B,))
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]
            x = self.flow_assemblies[i](x)
        return x, log_det_J

    def g(self, z: jt.Var, inj_f: List[jt.Var]) -> jt.Var:
        """INN inverse with injection subtraction。对应 deflow.py:183-189。"""
        for i in reversed(range(self.nflow_module)):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]
        return z

    # ----- 候选生成、选择与精修 helpers -----
    # 架构修正（2026-06-07，见 memory/patch-stitching-correspondence-constraint）：
    # 旧版 candidate 过生成 M->M*r 后子集选回 M 会重排槽位，破坏 patch_denoise stitching
    # 的"输出槽位 i == 输入点 i"不变量 → 全 arm 崩。本版改为 slot_variant：每个输入点 i
    # 独立生成 R 个变体，selector 只在 i 的 R 变体内选 1 个，输出第 i 槽恒为点 i 的精修，
    # 槽位语义保持，stitching 不受影响。
    def _gen_slot_variants(self, base, knn_k, R):
        """每个输入点 i 生成 R 个变体，输出 (B, M, R, 3)。

        variant 0 = base 点自身（identity 锚，保证"至少能退回 base"）。
        variant 1..R-1 = base[i] 与其第 j 个 KNN 近邻的中点 (base[i]+nbr_j)/2，
        天然落 i 的局部邻域、贴表面，且对 base 可微（梯度回流主干）。
        所有变体都"属于点 i"，selector 选哪个都仍是点 i 的位置 → 保槽位。
        """
        B, M, _ = base.shape
        if R <= 1:
            return base.unsqueeze(2)  # (B, M, 1, 3)
        # KNN（含自身，取 R 个近邻：第 0 个是自身，1..R-1 给中点）
        _, knn_idx = safe_knn(base, base, R)   # (B, M, R)
        bi = jt.arange(B).view(B, 1, 1).broadcast([B, M, R])
        nbr = base[bi, knn_idx]                    # (B, M, R, 3) 每点的 R 个近邻坐标
        base_e = base.unsqueeze(2).broadcast([B, M, R, 3])
        mids = 0.5 * (base_e + nbr)                # (B, M, R, 3) 中点变体
        # variant 0 用 base 自身（nbr[:,:,0] 是自身，中点=base，等价；显式覆盖更清晰）
        variants = jt.concat([base.unsqueeze(2), mids[:, :, 1:, :]], dim=2)  # (B,M,R,3)
        return variants

    def _select_variant_and_clean(self, variants, point_feat):
        """逐点在 R 个变体内选 1 个 + 可选 cleaner。返回 (B, M, 3)，第 i 槽 = 点 i 的精修。

        selector_mode:
          - identity: 固定选 variant 0（=base 自身）。对照臂 A/D，等价纯 base（候选 off 时）。
          - learned : MLP 对每点 R 变体打分 → 逐点 argmax 选 1（推理）。训练期 selector 由
                      system 的 per-slot R-way cross-entropy（clean-guided）监督；主 chamfer
                      走选中变体坐标，梯度经选中变体回流主干（不 detach）。
        cleaner_mode:
          - residual: selected += MLP_cleaner(concat(selected, point_feat))（zero-init→初始 identity）
        """
        B, M, R, _ = variants.shape
        bi = jt.arange(B).view(B, 1).broadcast([B, M])
        mi = jt.arange(M).view(1, M).broadcast([B, M])
        if self.selector_mode == "learned":
            # per-variant 特征 = 变体坐标(3) + 该点 predict_z(inj_channel) broadcast 到 R
            feat_e = point_feat.unsqueeze(2).broadcast([B, M, R, point_feat.shape[-1]])
            sel_in = jt.concat([variants, feat_e], dim=-1)     # (B, M, R, 3+inj)
            logit = self.MLP_selector(sel_in).squeeze(-1)      # (B, M, R)
            self._variant_logit = logit                        # 供 system 的 variant_ce loss
            var_idx, _ = jt.argmax(logit, dim=2)               # (B, M) 逐点选最高分变体
        else:  # identity
            self._variant_logit = None
            var_idx = jt.zeros((B, M)).int32()                 # 恒选 variant 0 = base
        self._selected_variant_idx = var_idx
        selected = variants[bi, mi, var_idx]                   # (B, M, 3) 对 base 可微，保槽位
        if self.cleaner_mode == "residual":
            clean_in = jt.concat([selected, point_feat], dim=-1)
            delta = self.MLP_cleaner(clean_in)                 # (B, M, 3) zero-init
            selected = selected + delta
        return selected

    def _fbm_topk_ste_mask(self) -> Tuple[jt.Var, jt.Var]:
        """返回固定预算的 STE 掩码及其硬前向掩码。"""
        assert self.fbm_mask_mode == "learnable_topk_ste"
        keep_count = self.pc_channel + self.aug_channel - self.cut_channel
        top_values, _ = jt.topk(
            self.fbm_mask_logit, k=keep_count, dim=-1, largest=True, sorted=True
        )
        threshold = top_values[..., -1:]
        hard_mask = (self.fbm_mask_logit >= threshold).float32().detach()
        soft_mask = jt.sigmoid(self.fbm_mask_logit)
        ste_mask = hard_mask + soft_mask - soft_mask.detach()
        return ste_mask, hard_mask

    # ----- Public interface -----
    def execute(self, xyz: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
        """Forward pass。

        Args:
            xyz: (B, N, pc_channel=3) noisy patch, already centered (减去 seed point)

        Returns:
            denoised: (B, N, pc_channel=3)
            ldj:      (B,) 当前基础方案恒 0
            loss_denoise: (,) FBM 分支恒 0，保留接口给 1.0 LBM/LCC
        """
        B, N, _ = xyz.shape

        # 1. Feature extraction (backbone-dependent)
        if self.backbone_mode == "mlgc":
            # KNN (shared between unit_coupling and feat_extract)
            # density/transport loss（2026-07-25）：改用 layer.safe_knn（jt.misc.knn 的
            # 带越界守卫副本）。原版内核 auto_parallel 尾块无线程守卫，
            # 会向 idx 缓冲之后野写、腐蚀本张量，predict 中间歇触发
            # cudaErrorIllegalAddress(700)（一次预测任务中曾连续触发）。
            # 数值语义与原版完全一致。
            _, knn_idx = safe_knn(xyz, xyz, self.num_neighbors)
            # injection features (12 个)
            inj_f = self.feat_extract(xyz, knn_idx)
            # aug feature
            aug = self.unit_coupling(xyz, knn_idx)

        elif self.backbone_mode == "group_token_hilbert":
            inj_f, aug = self.group_token_backbone(xyz)
            knn_idx = None  # not used by INN path

        x = jt.concat([xyz, aug], dim=-1)                # (B, N, pc + aug)

        # 3.5. 全局上下文模块 (hook between MLGC and INN)
        if self.global_context_mode == "hilbert_full":
            aug_global = self.global_block(aug, xyz)
            aug = aug + self.global_block.gamma * aug_global
            x = jt.concat([xyz, aug], dim=-1)
            # Stash diagnostics for logging
            self._global_diag = self.global_block.last_diag()

        # 4. INN forward with injection
        z, ldj = self.f(x, inj_f)

        # 5. FBM: 末尾 cut_channel 通道置零
        if self.fbm_mask_mode == "learnable_soft_tail":
            tail_keep = jt.sigmoid(self.fbm_tail_logit)
            predict_z = jt.concat([
                z[..., :-self.cut_channel],
                z[..., -self.cut_channel:] * tail_keep,
            ], dim=-1)
        elif self.fbm_mask_mode == "learnable_topk_ste":
            ste_mask, _ = self._fbm_topk_ste_mask()
            predict_z = z * ste_mask
        else:
            predict_z = z * self.channel_mask  # broadcast (1,1,C) -> (B,N,C)

        # 6. inverse 回点云空间 (implicit, log-only when direction head enabled)
        full_x = self.g(predict_z, inj_f)                 # (B, N, pc + aug)
        denoised_implicit = full_x[..., : self.pc_channel]  # (B, N, 3)

        # 6.5 post-FBM residual head
        if self.residual_head_mode == "post_fbm":
            head_input = jt.concat([denoised_implicit, predict_z], dim=-1)  # (B, N, 54)
            delta = self.residual_mlp(head_input)                            # (B, N, 3)
            # diagnostics (with no_grad for logging)
            with jt.no_grad():
                implicit_disp_norm = jt.sqrt(
                    ((denoised_implicit - xyz) ** 2).sum(dim=-1)
                ).mean()
                delta_norm = jt.sqrt((delta ** 2).sum(dim=-1)).mean()
                self._residual_delta_norm = float(delta_norm.item())
                self._residual_delta_ratio = float(
                    (delta_norm / (implicit_disp_norm + 1e-8)).item()
                )
            denoised = denoised_implicit + delta
        else:
            denoised = denoised_implicit

        # 7. direction-magnitude head (direction diagnostics and stabilization)
        if self.direction_head_mode in ("replace_output", "auxiliary"):
            if self.direction_head_feat_source == "predict_z_inj_last":
                head_feat = jt.concat([predict_z, inj_f[-1]], dim=-1)
            elif self.direction_head_feat_source == "z":
                head_feat = z
            else:
                head_feat = predict_z
            dir_raw = self.MLP_dir(head_feat)              # (B, N, 3)
            eps = 1e-4
            unit_dir = dir_raw / (jt.sqrt((dir_raw ** 2).sum(dim=-1, keepdims=True)) + eps)
            mag_raw = self.MLP_mag(head_feat)
            mag_raw_clamped = mag_raw.clamp(-20.0, 20.0)
            mag = jt.log(1.0 + jt.exp(mag_raw_clamped))  # softplus (B, N, 1)

            self._dh1_denoised_implicit = denoised_implicit
            self._dh1_head_unit_dir = unit_dir
            self._dh1_head_mag = mag

            if self.direction_head_mode == "auxiliary":
                if self.residual_head_mode == "off":
                    denoised = denoised_implicit
                # residual head 已设置 denoised，不覆盖
            else:
                denoised = xyz + mag * unit_dir           # replace_output: head 为主输出
        elif self.residual_head_mode == "off":
            denoised = denoised_implicit
            self._dh1_denoised_implicit = None
            self._dh1_head_unit_dir = None
            self._dh1_head_mag = None

        # 7.5 confidence-gated bounded pull.
        if self.pull_head_mode == "confidence_delta":
            assert self.MLP_pull is not None
            base_for_pull = denoised
            if self.pull_head_feat_source == "predict_z":
                pull_feat = predict_z
                input_tags = ["predict_z"]
            elif self.pull_head_feat_source == "z":
                pull_feat = z
                input_tags = ["z"]
            elif self.pull_head_feat_source == "denoised_implicit":
                pull_feat = denoised_implicit
                input_tags = ["denoised_implicit"]
            elif self.pull_head_feat_source == "base":
                pull_feat = base_for_pull
                input_tags = ["base"]
            elif self.pull_head_feat_source == "xyz":
                pull_feat = xyz
                input_tags = ["xyz"]
            else:
                pull_feat = jt.concat([predict_z, base_for_pull], dim=-1)
                input_tags = ["predict_z", "base"]
            pull_out = self.MLP_pull(pull_feat)
            pull_raw = pull_out[..., : self.pc_channel]
            confidence_logit = pull_out[..., self.pc_channel:]
            pull = jt.tanh(pull_raw) * self.pull_delta_max
            w = jt.sigmoid(confidence_logit)
            denoised = base_for_pull + w * pull
            self._pull_w = w
            self._pull_vector = w * pull
            self._pull_norm = jt.sqrt((self._pull_vector ** 2).sum(dim=-1) + 1e-12)
            self._pull_input_audit = {
                "pull_head_mode": self.pull_head_mode,
                "pull_head_feat_source": self.pull_head_feat_source,
                "allowed_inputs_only": True,
                "input_tags": input_tags,
                "forbidden_inputs": [],
            }
        else:
            self._pull_w = None
            self._pull_norm = None
            self._pull_vector = None
            self._pull_input_audit = {
                "pull_head_mode": self.pull_head_mode,
                "pull_head_feat_source": self.pull_head_feat_source,
                "allowed_inputs_only": True,
                "input_tags": [],
                "forbidden_inputs": [],
            }

        if self.keep_head_mode == "soft_weight":
            # Jittor note: use detach() instead of stop_grad() so keep losses
            # cannot backprop into the shared latent trunk, while keep-head
            # parameters themselves still receive gradients.
            self._keep_logit = self.MLP_keep(predict_z.detach())
        else:
            self._keep_logit = None

        # 8. 候选生成、选择与精修（INN 之后、patch 级）。
        # 每点 R 变体，selector 逐点选 1，输出第 i 槽恒 = 点 i 的精修 → 保 stitching 槽位
        # 对应（修正旧子集选重排破坏 stitching 的 bug，见 memory/patch-stitching-...）。
        # NOT detach —— 梯度经选中变体坐标与特征回流主干。off 时整段跳过、保持原输出不变。
        self._variant_logit = None
        self._selected_variant_idx = None
        self._slot_variants = None
        if self.candidate_mode == "slot_variant" and (
                self.selector_mode != "off" or self.cleaner_mode != "off"):
            variants = self._gen_slot_variants(
                denoised, self.candidate_knn_k, self.candidate_R)   # (B, M, R, 3)
            self._slot_variants = variants  # 供 system 的 variant_ce 算 target
            point_feat = predict_z  # (B, M, inj_channel) 带梯度，不 detach
            denoised = self._select_variant_and_clean(variants, point_feat)

        loss_denoise = jt.zeros((1,))                     # FBM 分支占位
        return denoised, ldj, loss_denoise

    # ----- 便捷接口 -----
    @jt.no_grad()
    def denoise(self, noisy_patch: jt.Var) -> jt.Var:
        """纯推理入口，只返回 denoised。"""
        self.eval()
        denoised, _, _ = self(noisy_patch)
        return denoised
