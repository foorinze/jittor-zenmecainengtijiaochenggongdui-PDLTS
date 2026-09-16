"""PDLTSLightSystem：PDLTS Light 的 DummySystem 扩展。

职责:
    1. 过滤可训练参数 (参数过滤规则): 排除 BN running_stats / ActNorm is_inited /
       AffineCoupling mask / FBM channel_mask, 只把真正需要学的参数塞给
       optimizer. 参数过滤只作用于本系统，VMSystem 使用父类实现。
    2. 日志契约: 训练启动时在
       outputs/runs/<stage>/<run_id>/ 下创建 run 目录, 写入 command.sh / config.yaml /
       env.txt / git_commit.txt / manifest.json; 训练过程中按每 epoch 写
       logs/epoch_summary.jsonl, 按每 step 写 logs/train.log 的 mean loss 行.
    3. Checkpoint 保真: 训练结束后 ckpt 里必须同时保存 channel_mask 和所有
       ActNorm is_inited. 继承 DummySystem 默认 save 已经走 model.save (jt.save
       module 会 save 所有 parameters, 包含状态变量), 不额外改.
    4. 不改变训练循环的核心顺序: forward -> zero_grad -> backward -> step.

口径:
    - run_id 默认由构造参数 `run_id` 指定, 不自动生成 (方便可复现).
    - 如果未提供 `run_id`, 使用时间戳 `YYYYMMDD_HHMMSS_<run_tag>`;
      公开 run_tag 建议采用 `a_final_<stage>_<scope>` 这类语义名。
    - 日志目录基底由构造参数 `run_output_root` 指定, 默认 `../outputs/runs`.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import jittor as jt
import numpy as np
from jittor import optim
from tqdm import tqdm

from .spec import DummySystem, DummyWriter, _get_item


def _infer_output_version(run_id: str, run_tag: str = "") -> str:
    """从 run_id/run_tag 推断公开榜单阶段。"""
    text = f"{run_id} {run_tag}"
    for public_version in ("a_final", "b_final"):
        if re.search(r"(^|_)" + re.escape(public_version) + r"($|_)", text):
            return public_version
    return "_unversioned"


def _is_managed_output_root(root: str) -> bool:
    norm = os.path.normpath(str(root)).replace("\\", "/")
    return (
        norm in {"outputs/runs", "outputs/predictions"}
        or norm.endswith("/outputs/runs")
        or norm.endswith("/outputs/predictions")
    )


def _normalize_config_version(config_version: Optional[str]) -> str:
    if config_version is None or str(config_version).strip() == "":
        return ""
    version = str(config_version).strip()
    if version in {"a_final", "b_final"}:
        return version
    raise ValueError(f"invalid CONFIG_VERSION for public output layout: {version!r}")


def _versioned_output_dir(
    root: str,
    run_id: str,
    run_tag: str = "",
    config_version: Optional[str] = None,
) -> str:
    version = _infer_output_version(run_id, run_tag)
    fallback_version = _normalize_config_version(config_version)
    if version == "_unversioned" and fallback_version:
        version = fallback_version
    if version == "_unversioned" and _is_managed_output_root(root):
        raise ValueError(
            "cannot infer output version for managed outputs root; "
            "include a public stage such as a_final in run_id/run_tag or set CONFIG_VERSION"
        )
    return os.path.join(root, version, run_id)


# ---------------------------------------------------------------------------
# 可训练参数过滤: 排除状态变量, 只返回可训练参数列表.
# 排除运行统计量、初始化标志和固定掩码。
# ---------------------------------------------------------------------------
def is_trainable_param_name(name: str) -> bool:
    """判断一个 named_parameters() 的 name 是否是可训练参数 (应进 optimizer)."""
    # Batch/Sync norm 的运行统计量 (running_mean / running_var)
    if name.endswith("running_mean") or name.endswith("running_var"):
        return False
    # ActNorm1d 的 first-forward init 标志
    if name.endswith("is_inited"):
        return False
    # AffineCoupling 的固定 mask (.mask) 和整网 FBM channel_mask.
    # 裸 PDLTSLightNetwork 里叫 channel_mask, 包进 PDLTSLight(ModelSpec) 后叫 network.channel_mask.
    if name.endswith(".mask") or name.endswith("channel_mask"):
        return False
    return True


def _normalize_trainable_param_prefixes(
    trainable_param_prefixes: Optional[List[str]] = None,
) -> Optional[List[str]]:
    if trainable_param_prefixes is None:
        return None
    prefixes = [str(p) for p in trainable_param_prefixes if str(p)]
    if not prefixes:
        raise RuntimeError("trainable_param_prefixes was provided but is empty")
    return prefixes


def collect_trainable_named_params(
    model,
    trainable_param_prefixes: Optional[List[str]] = None,
) -> List[tuple[str, jt.Var]]:
    """过滤非训练状态变量, 可选再按前缀白名单冻结 trunk."""
    prefixes = _normalize_trainable_param_prefixes(trainable_param_prefixes)
    named_params: List[tuple[str, jt.Var]] = []
    for name, p in model.named_parameters():
        if not is_trainable_param_name(name):
            continue
        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        named_params.append((name, p))
    return named_params


def collect_trainable_params(
    model,
    trainable_param_prefixes: Optional[List[str]] = None,
) -> List[jt.Var]:
    """过滤非训练状态变量, 返回应进 optimizer 的参数列表."""
    return [
        p
        for _name, p in collect_trainable_named_params(
            model,
            trainable_param_prefixes=trainable_param_prefixes,
        )
    ]


def make_filtered_optimizer(
    optimizer_config: dict,
    model,
    trainable_param_prefixes: Optional[List[str]] = None,
):
    """过滤非训练状态变量参数后构建 optimizer.

    不使用 DummySystem.spec.get_optimizer (它会调 model.parameters() 无过滤).
    """
    cfg = dict(optimizer_config)  # 不动 caller
    __target__ = cfg.pop("__target__")
    MAPPING = {"sgd": optim.SGD, "adam": optim.Adam}
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    trainable_named = collect_trainable_named_params(
        model,
        trainable_param_prefixes=trainable_param_prefixes,
    )
    trainable = [p for _name, p in trainable_named]
    if not trainable:
        raise RuntimeError(
            "no trainable params found after state/prefix filtering "
            f"(trainable_param_prefixes={trainable_param_prefixes})"
        )
    return OptimizerClass(trainable, **cfg)


def _optimizer_state_dict(optimizer) -> Dict[str, Any]:
    """Return a serializable optimizer state or fail loudly.

    Strict resume is only honest when optimizer state can be saved and loaded.
    Jittor optimizer APIs have varied across versions, so this helper probes the
    runtime object instead of assuming a PyTorch-compatible interface.
    """
    if hasattr(optimizer, "state_dict") and callable(optimizer.state_dict):
        state = optimizer.state_dict()
        if state is None:
            raise RuntimeError("optimizer.state_dict() returned None")
        return state
    raise RuntimeError(
        "optimizer does not expose state_dict(); this run cannot produce a "
        "strict-resume training package"
    )


def _load_optimizer_state_dict(optimizer, state: Dict[str, Any]):
    if hasattr(optimizer, "load_state_dict") and callable(optimizer.load_state_dict):
        optimizer.load_state_dict(state)
        return
    raise RuntimeError(
        "optimizer does not expose load_state_dict(); cannot strict-resume "
        "from this training package"
    )


def _train_state_path(ckpt_path: str) -> str:
    root, ext = os.path.splitext(ckpt_path)
    if ext:
        return f"{root}.train{ext}"
    return f"{ckpt_path}.train.pkl"


def _save_training_state(
    *,
    path: str,
    model_ckpt_path: str,
    optimizer,
    epoch: int,
    global_step: int,
    start_epoch: int,
    run_id: str,
    run_dir: str,
    ckpt_save_name: str,
    trainer_config: Optional[dict],
    optimizer_config: Optional[dict],
    loss_config: Optional[dict],
):
    state = {
        "format": "pdlts_train_state_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "global_step": global_step,
        "start_epoch": start_epoch,
        "run_id": run_id,
        "run_dir": run_dir,
        "ckpt_save_name": ckpt_save_name,
        "model_ckpt_path": model_ckpt_path,
        "optimizer_state": _optimizer_state_dict(optimizer),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "jittor_seed": jt.get_seed() if hasattr(jt, "get_seed") else None,
        },
        "config": {
            "trainer_config": trainer_config,
            "optimizer_config": optimizer_config,
            "loss_config": loss_config,
        },
    }
    jt.save(state, path)


def _load_training_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume_state not found: {path}")
    state = jt.load(path)
    if not isinstance(state, dict) or state.get("format") != "pdlts_train_state_v1":
        raise ValueError(
            f"not a pdlts strict training state package: {path}; use load_ckpt "
            "for model-only warm-start"
        )
    if "optimizer_state" not in state:
        raise ValueError(f"resume_state has no optimizer_state: {path}")
    if "model_ckpt_path" not in state:
        raise ValueError(f"resume_state has no model_ckpt_path: {path}")
    return state


def _resolve_resume_model_path(resume_state_path: str, state: Dict[str, Any]) -> str:
    model_path = state["model_ckpt_path"]
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    if os.path.exists(model_path):
        return model_path
    candidate = os.path.join(os.path.dirname(resume_state_path), os.path.basename(model_path))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"model checkpoint referenced by resume_state is missing: {model_path}"
    )


# ---------------------------------------------------------------------------
# Run 目录 / 日志契约
# ---------------------------------------------------------------------------
def _try_git_status() -> dict:
    """返回 git commit / branch / dirty. 不是 git 仓库时返回 status=not-a-repo.

    在跨文件系统边界或大型工作区中，`git status --porcelain` 可能耗时数秒，并随
    outputs/ 产物增多继续上升。原 5 秒超时会被 subprocess.run 的内部
    kill() 命中，而 Jittor 装了进程级 SIGCHLD handler，把"任意子进程被杀"
    无差别误判为 OOM 并 quick_exit 整个训练/预测进程——即使被杀的只是一次无关
    的 git 查询。30 秒是一次性 run 启动成本，不影响训练/预测循环本身耗时。
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if commit.returncode != 0:
            return {"status": "not-a-repo"}
        commit_hash = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip() or "UNKNOWN"
        dirty_probe = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        dirty = "yes" if dirty_probe.stdout.strip() else "no"
        return {"status": "ok", "commit": commit_hash, "branch": branch, "dirty": dirty}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _write_git_commit(run_dir: str):
    info = _try_git_status()
    path = os.path.join(run_dir, "git_commit.txt")
    with open(path, "w", encoding="utf-8") as f:
        if info["status"] == "ok":
            f.write(f"Commit: {info['commit']}\n")
            f.write(f"Branch: {info['branch']}\n")
            f.write(f"Dirty: {info['dirty']}\n")
        else:
            f.write(f"status: {info['status']}\n")
            if "error" in info:
                f.write(f"error: {info['error']}\n")


def _write_env_txt(run_dir: str):
    path = os.path.join(run_dir, "env.txt")
    lines: List[str] = []
    lines.append(f"python: {sys.version.splitlines()[0]}\n")
    try:
        lines.append(f"jittor: {jt.__version__}\n")
        lines.append(f"use_cuda: {jt.flags.use_cuda}\n")
    except Exception:
        pass
    try:
        import numpy as np
        lines.append(f"numpy: {np.__version__}\n")
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_command_sh(run_dir: str, argv: Optional[List[str]] = None):
    """记录本次运行的命令 (argv 默认取 sys.argv)."""
    cmd = argv if argv is not None else sys.argv
    path = os.path.join(run_dir, "command.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# auto-generated by PDLTSLightSystem at run start\n")
        f.write(" ".join(repr(s) if " " in s else s for s in cmd) + "\n")


def _write_config_snapshot(run_dir: str, payload: dict):
    """把 loss/optimizer/trainer/ckpt 配置快照落盘."""
    path = os.path.join(run_dir, "config.yaml")
    # 用 json 转 yaml 可读近似; 依赖 omegaconf 可能会解析副作用, 直接 json dump 即可
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


def _write_resume_snapshot(run_dir: str, payload: dict):
    """续训时保留原始 config.yaml, 另写一份 resume 快照。"""
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"resume_{stamp}.json")
    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        **payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    return path


def _write_manifest(run_dir: str, extra: Optional[dict] = None):
    """初始化 manifest.json, 后续 on_train_epoch_end 会追写 summary."""
    path = os.path.join(run_dir, "manifest.json")
    data = {
        "run_id": os.path.basename(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "train",
        "summary": {},
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _write_notes_md(run_dir: str):
    """创建运行记录 notes.md，已有文件保持不变。"""
    path = os.path.join(run_dir, "notes.md")
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Notes\n\n")
        f.write("- status: auto-created at run start\n")
        f.write("- conclusion: pending\n")


# ---------------------------------------------------------------------------
# PDLTSLightWriter: 推理写入 denoised.npy, 路径去掉 input_dataset_dir 前缀
# ---------------------------------------------------------------------------
class PDLTSLightWriter(DummyWriter):
    """把 prediction['pc_denoised'] 写到 <save_dir>/<asset 相对路径>/<save_name>.npy.

    关键: asset.path 是 "<input_dataset_dir>/shapenet/<cls>/<id>/<data_name>"
    (绝对路径或相对路径), 提交时需要 "shapenet/<cls>/<id>/denoised.npy" 目录层级.
    所以 writer 要按 asset 相对于 input_dataset_dir 的路径拼, 而不是直接走
    os.path.dirname(asset.path) (那会把 `../dataset/test_noisy/` 前缀也带进来).

    依赖数据层 PCDataModule 提供 predict_datapath 字典 (dataset_module.predict_datapath),
    但 write() 参数签名里 dataset_module 是可选的, 所以这里直接从 asset.path 截除
    data_name (noisy.npy) 这部分, 保留父目录 dirname(path) 然后截 shapenet/ 之前
    的前缀.

    实现: 找 "shapenet/" 在 path 里的位置, 从那开始截取. 赛题目录结构以 "shapenet/"
    为根, 这是赛题契约里明确的.
    """

    def __init__(self, save_dir: str = "predictions", save_name: str = "denoised"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name

    def write(self, batch, prediction: List[Dict], dataset_module=None):
        for i, asset in enumerate(batch["asset"]):
            path = asset.path
            assert path is not None, "asset.path is None, cannot write"

            # Windows 风格 \ 在 Linux / WSL 下不被 os.path.dirname 识别为分隔符.
            # 先正规化成 / 再走截取逻辑.
            path_norm = path.replace("\\", "/")
            dirname = os.path.dirname(path_norm)

            # 必须找到 "shapenet/" 作为根锚点; 找不到就 fail-loudly,
            # 不走静默 fallback, 路径异常直接报错.
            anchor = "shapenet/"
            if anchor not in dirname:
                raise AssertionError(
                    f"PDLTSLightWriter: cannot find '{anchor}' anchor in asset.path "
                    f"dirname='{dirname}' (original path='{path}'). "
                    f"This usually means datalist / loader changed the directory "
                    f"layout. Refusing to write to an unexpected location."
                )
            rel = dirname[dirname.index(anchor):]
            # rel 此时形如 "shapenet/<cls>/<id>"

            out_dir = os.path.join(self.save_dir, rel)
            # 防御: 绝对禁止写回 dataset/ (用 commonpath 替代
            # startswith, 避免 "/foo/dataset_x" 前缀 "/foo/dataset" 的误判).
            abs_out = os.path.abspath(out_dir)
            abs_dataset = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
            )
            try:
                common = os.path.commonpath([abs_out, abs_dataset])
            except ValueError:
                # 跨盘符 (Windows 独有) 会抛 ValueError, 说明肯定不在 dataset 里.
                common = ""
            assert common != abs_dataset, (
                f"refusing to write prediction into dataset/: abs_out={abs_out}"
            )

            os.makedirs(out_dir, exist_ok=True)

            denoised = prediction[i]["pc_denoised"]
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            # shape guard: 与 noisy 输入点数一致
            noisy_np = asset.sampled_vertices_noisy
            if noisy_np is not None:
                assert denoised_np.shape == noisy_np.shape, (
                    f"shape mismatch before write: denoised={denoised_np.shape}, "
                    f"noisy={noisy_np.shape}, path={path}"
                )
            np.save(os.path.join(out_dir, f"{self.save_name}.npy"),
                    denoised_np.astype(np.float32))
            for sidecar_key, filename, expected_dim in (
                ("confidence_w", "confidence_w.npy", 1),
                ("pull_norm", "pull_norm.npy", 1),
                ("pull_vector", "pull_vector.npy", 2),
            ):
                if sidecar_key not in prediction[i]:
                    continue
                sidecar = prediction[i][sidecar_key]
                sidecar_np = sidecar if isinstance(sidecar, np.ndarray) else sidecar.numpy()
                if expected_dim == 1:
                    assert sidecar_np.shape == (denoised_np.shape[0],), (
                        f"{sidecar_key} shape mismatch before write: "
                        f"{sidecar_np.shape}, denoised={denoised_np.shape}, path={path}"
                    )
                else:
                    assert sidecar_np.shape == denoised_np.shape, (
                        f"{sidecar_key} shape mismatch before write: "
                        f"{sidecar_np.shape}, denoised={denoised_np.shape}, path={path}"
                    )
                assert np.isfinite(sidecar_np).all(), (
                    f"{sidecar_key} has NaN/Inf before write: path={path}"
                )
                np.save(os.path.join(out_dir, filename), sidecar_np.astype(np.float32))


# ---------------------------------------------------------------------------
# PDLTSLightSystem
# ---------------------------------------------------------------------------
class PDLTSLightSystem(DummySystem):
    """PDLTS Light 的训练 system. 继承 DummySystem, 替换 optimizer 构建 + 加日志契约.

    新增构造参数:
        run_id: run 目录名. 未指定时自动生成
            `YYYYMMDD_HHMMSS_<run_tag>`, 新 run_tag 建议采用
            `a_final_<stage>_<scope>_ep<N>`.
        run_output_root: 顶级 run 目录, 默认 `../outputs/runs` (从 starter_code/
            执行时指向 PDLTS/outputs/runs).
    """

    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
        run_id: Optional[str] = None,
        run_output_root: str = "../outputs/runs",
        run_tag: str = "a_final",
        load_ckpt: Optional[str] = None,
        start_epoch: int = 0,
        resume_state: Optional[str] = None,
        resume_run: bool = False,
        CONFIG_VERSION: Optional[str] = None,
        trainable_param_prefixes: Optional[List[str]] = None,
    ):
        # 不走父类的 optimizer 构建 (那会吃掉整 model.parameters 不做 参数过滤).
        # 先手动持有 optimizer_config, 后面自己建.
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get("epochs", 1)
        self._validation_loss = defaultdict(list)
        self._trainer_config = trainer_config
        self._optimizer_config = optimizer_config
        self._trainable_param_prefixes = _normalize_trainable_param_prefixes(
            trainable_param_prefixes
        )
        self._resume_state_path = resume_state
        self._resume_state_payload = None
        self._resume_model_ckpt = None
        if resume_state is not None:
            if load_ckpt is not None:
                raise ValueError("resume_state and load_ckpt are mutually exclusive")
            self._resume_state_payload = _load_training_state(resume_state)
            self._resume_model_ckpt = _resolve_resume_model_path(
                resume_state, self._resume_state_payload
            )
            start_epoch = int(self._resume_state_payload["next_epoch"])
            print(
                "[PDLTSLightSystem] strict-resume from "
                f"{resume_state} (model={self._resume_model_ckpt}, "
                f"next_epoch={start_epoch})"
            )

        # warm-start: 加载已有 ckpt 的模型权重（不含 optimizer state）
        if load_ckpt is not None and model is not None:
            print(f"[PDLTSLightSystem] warm-start from {load_ckpt}")
            model.load(load_ckpt)
        if self._resume_model_ckpt is not None and model is not None:
            model.load(self._resume_model_ckpt)
        self._start_epoch = start_epoch  # 绝对 epoch 偏移（用于 ckpt 命名）

        self._trainable_param_names: List[str] = []
        if model is not None:
            self._trainable_param_names = [
                name
                for name, _p in collect_trainable_named_params(
                    model,
                    trainable_param_prefixes=self._trainable_param_prefixes,
                )
            ]
        self._trainable_param_name_set = set(self._trainable_param_names)
        self._frozen_param_state: Dict[str, np.ndarray] = {}
        if model is not None and self._trainable_param_prefixes is not None:
            for name, p in model.named_parameters():
                if name not in self._trainable_param_name_set:
                    self._frozen_param_state[name] = p.numpy().copy()

        # 参数过滤: 只把可训练参数塞进 optimizer.
        if optimizer_config is not None and model is not None:
            self.optimizer = make_filtered_optimizer(
                optimizer_config,
                model,
                trainable_param_prefixes=self._trainable_param_prefixes,
            )
        else:
            self.optimizer = None
        if self._resume_state_payload is not None:
            if self.optimizer is None:
                raise RuntimeError("strict resume requires optimizer_config")
            _load_optimizer_state_dict(
                self.optimizer, self._resume_state_payload["optimizer_state"]
            )
            rng_state = self._resume_state_payload.get("rng_state", {})
            try:
                if "python" in rng_state:
                    random.setstate(rng_state["python"])
                if "numpy" in rng_state:
                    np.random.set_state(rng_state["numpy"])
                if rng_state.get("jittor_seed") is not None and hasattr(jt, "set_seed"):
                    jt.set_seed(int(rng_state["jittor_seed"]))
            except Exception as e:
                raise RuntimeError(f"failed to restore RNG state for strict resume: {e}")
            self._global_step_counter = int(self._resume_state_payload.get("global_step", 0))

        # ---- 日志契约准备 ----
        resume_run = bool(resume_run)
        if run_id is None:
            assert not resume_run, "resume_run=True requires an explicit run_id"
            run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{run_tag}"
        self.run_id = run_id
        self.run_dir = _versioned_output_dir(
            run_output_root, run_id, run_tag, CONFIG_VERSION
        )
        if resume_run and not os.path.isdir(self.run_dir):
            raise FileNotFoundError(
                f"resume_run=True but run_dir does not exist: {self.run_dir}"
            )
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        # 快照 config + env + command + git
        snapshot_payload = {
            "loss_config": loss_config,
            "optimizer_config": optimizer_config,
            "trainer_config": trainer_config,
            "ckpt_save_dir": ckpt_save_dir,
            "ckpt_save_name": ckpt_save_name,
            "run_id": run_id,
            "run_output_root": run_output_root,
            "run_tag": run_tag,
            "load_ckpt": load_ckpt,
            "start_epoch": start_epoch,
            "resume_state": resume_state,
            "resume_mode": "strict" if resume_state else ("warm_start" if load_ckpt else "none"),
            "resume_run": resume_run,
            "trainable_param_prefixes": self._trainable_param_prefixes,
            "trainable_param_count": len(self._trainable_param_names),
            "trainable_param_names": self._trainable_param_names,
            "CONFIG_VERSION": CONFIG_VERSION,
        }
        if resume_run:
            _write_resume_snapshot(self.run_dir, snapshot_payload)
            self.manifest_path = os.path.join(self.run_dir, "manifest.json")
            if not os.path.exists(self.manifest_path):
                self.manifest_path = _write_manifest(self.run_dir, extra={"resume_run": True})
        else:
            _write_command_sh(self.run_dir)
            _write_env_txt(self.run_dir)
            _write_git_commit(self.run_dir)
            _write_config_snapshot(self.run_dir, snapshot_payload)
            self.manifest_path = _write_manifest(
                self.run_dir,
                extra={
                    "trainable_param_filter": {
                        "prefixes": self._trainable_param_prefixes,
                        "param_count": len(self._trainable_param_names),
                        "param_names": self._trainable_param_names,
                        "frozen_param_restore_enabled": bool(self._frozen_param_state),
                        "frozen_param_restore_count": len(self._frozen_param_state),
                    }
                },
            )

        # 日志文件 handle
        self._log_train_path = os.path.join(self.run_dir, "logs", "train.log")
        self._log_epoch_path = os.path.join(self.run_dir, "logs", "epoch_summary.jsonl")
        self._metrics_csv_path = os.path.join(self.run_dir, "logs", "metrics.csv")
        # truncate (新 run 一律新日志)
        if resume_run:
            open(self._log_train_path, "a", encoding="utf-8").close()
            open(self._log_epoch_path, "a", encoding="utf-8").close()
            if (not os.path.exists(self._metrics_csv_path)) or os.path.getsize(self._metrics_csv_path) == 0:
                with open(self._metrics_csv_path, "w", encoding="utf-8") as f:
                    f.write("epoch,loss_sum_mean,loss_sum_count\n")
        else:
            open(self._log_train_path, "w").close()
            open(self._log_epoch_path, "w").close()
            with open(self._metrics_csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,loss_sum_mean,loss_sum_count\n")
        _write_notes_md(self.run_dir)
        if resume_run:
            self._log_line(
                f"[resume] mode={snapshot_payload['resume_mode']} "
                f"start_epoch={start_epoch} epochs={self.epochs} "
                f"load_ckpt={load_ckpt} resume_state={resume_state}"
            )
        # 训练统计缓冲 (每 epoch 重置)
        self._epoch_loss_history: Dict[str, list] = defaultdict(list)
        self._current_epoch = -1
        self._current_step = 0
        # Part A: per-step metrics sampling interval (1 = every step, N = every N steps)
        self._metrics_sample_interval = max(1, int(trainer_config.get("metrics_sample_interval", 1)))
        # Default to the 1.x-style terminal UX: one live tqdm bar per epoch.
        # The Gate wrapper must inherit the real TTY; piping/teeing stdout turns
        # tqdm carriage-return refreshes into noisy line-by-line output.
        self._progress_bar_mode = str(trainer_config.get("progress_bar", "epoch")).lower()
        if not hasattr(self, "_global_step_counter"):
            self._global_step_counter = 0

    # ---- helper ----
    def _log_line(self, line: str):
        with open(self._log_train_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_strict_resume_state(self, ckpt_path: str, abs_ep: int):
        train_state_path = _train_state_path(ckpt_path)
        _save_training_state(
            path=train_state_path,
            model_ckpt_path=ckpt_path,
            optimizer=self.optimizer,
            epoch=abs_ep,
            global_step=self._global_step_counter,
            start_epoch=self._start_epoch,
            run_id=self.run_id,
            run_dir=self.run_dir,
            ckpt_save_name=self.ckpt_save_name,
            trainer_config=self._trainer_config,
            optimizer_config=self._optimizer_config,
            loss_config=self.loss_config,
        )
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"run_id": self.run_id, "summary": {}}
        data["strict_resume"] = {
            "available": True,
            "latest_epoch": abs_ep,
            "latest_model_ckpt": ckpt_path,
            "latest_train_state": train_state_path,
            "note": "Use resume_state, not load_ckpt, for strict continuation.",
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._log_line(f"[checkpoint] model={ckpt_path} train_state={train_state_path}")
        return train_state_path

    # ---- 重写少量 hook 记录日志 ----
    def on_train_epoch_start(self):
        self._current_epoch += 1
        self._current_step = 0
        self._epoch_loss_history.clear()
        # L2 target rescue: 把当前 epoch 推给 model, 供
        # pred_nn_clean warmup 判断 (model.training_step 不知道自己是第几个 epoch).
        # 用 absolute epoch (start_epoch + current) 与 ckpt 命名 (line ~452 的 abs_ep)
        # 同口径; 这样 resume 训练时 warmup 不会被重置.
        # hasattr guard 保持对未实装 set_train_epoch 的旧 model 的兼容.
        if hasattr(self.model, "set_train_epoch"):
            abs_ep = self._start_epoch + self._current_epoch
            self.model.set_train_epoch(abs_ep)
        self._log_line(f"[epoch {self._start_epoch + self._current_epoch}] start")

    def _restore_frozen_param_state(self) -> int:
        """Restore non-whitelisted params/state for prefix-frozen forensic runs.

        Jittor modules can update BatchNorm running stats in train mode even when
        the optimizer only owns whitelisted parameters.  For frozen-trunk
        forensics, the persisted checkpoint must keep every non-whitelisted
        parameter/state byte-identical to the warm-start snapshot.
        """
        if not self._frozen_param_state or self.model is None:
            return 0
        by_name = {name: p for name, p in self.model.named_parameters()}
        restored = 0
        for name, arr in self._frozen_param_state.items():
            p = by_name.get(name)
            if p is None:
                raise RuntimeError(f"frozen param missing during restore: {name}")
            p.assign(jt.array(arr))
            restored += 1
        return restored

    def training_step(self, batch):
        """继承父类 forward, 但额外把每个 loss 项记进本 epoch 统计."""
        loss_sum = self.forward(batch, validate=False)

        should_sample = (self._global_step_counter % self._metrics_sample_interval == 0)
        if should_sample:
            loss_item = _get_item(loss_sum)
            self._epoch_loss_history["loss_sum"].append(loss_item)

            last_metrics = getattr(self.model, "_last_train_metrics", None) or {}
            for name, val in last_metrics.items():
                try:
                    val_float = _get_item(val) if isinstance(val, jt.Var) else float(val)
                    self._epoch_loss_history[name].append(val_float)
                except (TypeError, ValueError):
                    pass

            self._log_line(
                f"[epoch {self._start_epoch + self._current_epoch}] step {self._current_step} "
                f"loss_sum={loss_item}"
            )
        self._current_step += 1
        self._global_step_counter += 1
        return loss_sum

    def on_train_epoch_end(self):
        ep = self._start_epoch + self._current_epoch
        summary = {
            "epoch": ep,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for name, vals in self._epoch_loss_history.items():
            if not vals:
                continue
            summary[f"{name}_mean"] = sum(vals) / len(vals)
            summary[f"{name}_count"] = len(vals)
        extra_summary = getattr(self, "_epoch_extra_summary", None)
        if isinstance(extra_summary, dict):
            summary.update(extra_summary)
        with open(self._log_epoch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        with open(self._metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{summary.get('loss_sum_mean')},"
                f"{summary.get('loss_sum_count')}\n"
            )
        self._last_epoch_summary = summary
        self._log_line(f"[epoch {ep}] end  "
                       f"loss_sum_mean={summary.get('loss_sum_mean')!r}")

        # 更新 manifest 的 summary
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"run_id": self.run_id, "summary": {}}
        summary_block = data.get("summary", {})
        epoch_block = {
            "loss_sum_mean": summary.get("loss_sum_mean"),
        }
        if isinstance(extra_summary, dict):
            epoch_block.update(extra_summary)
        summary_block[f"epoch_{ep}"] = epoch_block
        summary_block["last_epoch"] = ep
        data["summary"] = summary_block
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._epoch_extra_summary = None

    # ---- ckpt 路径改到 run_dir/checkpoints, 否则父类写 `experiments/...` ----
    def train(self):
        """复用 DummySystem.train 的主循环, 但:
        - ckpt 写到 run_dir/checkpoints/
        - ckpt 命名用绝对 epoch (self._start_epoch + epoch)
        """
        self.ckpt_save_dir = os.path.join(self.run_dir, "checkpoints")
        self.model.set_predict(False)
        assert self.optimizer is not None, "optimizer is None, cannot train"

        for epoch in range(self.epochs):
            abs_ep = self._start_epoch + epoch
            self.model.train()
            self.on_train_epoch_start()
            train_dl = self.dataset_module.train_dataloader()
            assert train_dl is not None
            progress_off = self._progress_bar_mode in {"off", "none", "false", "0", "silent"}
            line_only = self._progress_bar_mode in {"line", "epoch_line"}
            show_epoch_bar = not progress_off and not line_only
            iterator = train_dl
            if show_epoch_bar:
                iterator = tqdm(
                    train_dl,
                    total=len(train_dl) // train_dl.batch_size,
                    desc=f"Epoch {abs_ep}/{self._start_epoch + self.epochs - 1}",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    leave=True,
                )
            batch_count = 0
            last_loss = None
            for batch_idx, batch in enumerate(iterator):
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                last_loss = _get_item(loss)
                if show_epoch_bar and batch_idx % self._metrics_sample_interval == 0:
                    iterator.set_postfix(loss=f"{last_loss:.6g}", refresh=False)
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
                batch_count += 1
            self.on_train_epoch_end()
            if line_only:
                summary = getattr(self, "_last_epoch_summary", {}) or {}
                print(
                    f"Epoch {abs_ep}/{self._start_epoch + self.epochs - 1}, "
                    f"batches={batch_count}, "
                    f"loss_sum_mean={summary.get('loss_sum_mean')!r}, "
                    f"last_loss={last_loss!r}"
                )

            restored_count = self._restore_frozen_param_state()
            if restored_count:
                self._log_line(
                    f"[epoch {abs_ep}] restored {restored_count} frozen params/states before validation/save"
                )
            self.model.eval()
            val_dl = self.dataset_module.validate_dataloader()
            if val_dl is not None:
                self.on_validation_epoch_start()
                if isinstance(val_dl, dict):
                    for name, dl in val_dl.items():
                        for batch in tqdm(dl, total=len(dl) // dl.batch_size,
                                          desc=f"Epoch {abs_ep}, Validate {name}"):
                            self.on_validation_batch_start()
                            vloss = self.validation_step(batch)
                            self.on_validation_batch_end()
                else:
                    for batch in tqdm(val_dl, total=len(val_dl) // val_dl.batch_size,
                                      desc=f"Epoch {abs_ep}, Validate"):
                        self.on_validation_batch_start()
                        vloss = self.validation_step(batch)
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()

            ckpt_path = os.path.join(self.ckpt_save_dir,
                                     f"{self.ckpt_save_name}_{abs_ep}.pkl")
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self._restore_frozen_param_state()
            self.model.save(ckpt_path)
            self._save_strict_resume_state(ckpt_path, abs_ep)


class PDLTSLightShardRotationSystem(PDLTSLightSystem):
    """在一个 macro epoch 内轮转固定 1K 级 shard。

    模型、loss、optimizer、validation、checkpoint 逻辑保持和 PDLTSLightSystem 对齐；
    唯一改变是每个内部 shard pass 前切换 train datalist。
    """

    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
        run_id: Optional[str] = None,
        run_output_root: str = "../outputs/runs",
        run_tag: str = "a_final_fixed_shard",
        load_ckpt: Optional[str] = None,
        start_epoch: int = 0,
        resume_state: Optional[str] = None,
        shard_datalist_dir: str = "",
        num_shards: int = 16,
        fixed_shard_order: bool = True,
        source_datalist: str = "",
        shard_seed: int = 17016,
    ):
        if not shard_datalist_dir:
            raise ValueError("shard_datalist_dir is required for shard rotation")
        self.shard_datalist_dir = shard_datalist_dir
        self.num_shards = int(num_shards)
        self.fixed_shard_order = bool(fixed_shard_order)
        self.source_datalist = source_datalist
        self.shard_seed = int(shard_seed)
        self._shards: List[Dict] = []
        self._shard_meta: Dict = {}
        self._shard_schedule: List[Dict] = []

        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
            run_id=run_id,
            run_output_root=run_output_root,
            run_tag=run_tag,
            load_ckpt=load_ckpt,
            start_epoch=start_epoch,
            resume_state=resume_state,
        )

        self._shards, self._shard_meta = self._load_and_verify_shards()
        self._shard_schedule_path = os.path.join(
            self.run_dir, "logs", "shard_schedule.json"
        )
        self._update_run_artifacts_for_shards()
        self._write_shard_schedule()

    @staticmethod
    def _read_datalist(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _load_and_verify_shards(self):
        shards: List[Dict] = []
        shard_root = os.path.abspath(self.shard_datalist_dir)
        for shard_id in range(self.num_shards):
            path = os.path.join(shard_root, f"shard_{shard_id:02d}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing shard datalist: {path}")
            lines = self._read_datalist(path)
            if not lines:
                raise ValueError(f"empty shard datalist: {path}")
            shards.append({
                "id": shard_id,
                "path": path,
                "lines": lines,
                "size": len(lines),
            })

        all_lines: List[str] = []
        for shard in shards:
            all_lines.extend(shard["lines"])
        duplicate_count = len(all_lines) - len(set(all_lines))
        if duplicate_count != 0:
            raise ValueError(
                f"fixed shard datalists have duplicated samples: {duplicate_count}"
            )

        source_match = None
        source_count = None
        if self.source_datalist:
            source_path = os.path.abspath(self.source_datalist)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"missing source datalist: {source_path}")
            source_lines = self._read_datalist(source_path)
            source_count = len(source_lines)
            source_match = sorted(all_lines) == sorted(source_lines)
            if not source_match:
                raise ValueError(
                    "fixed shard union does not match source_datalist: "
                    f"union={len(all_lines)}, source={source_count}"
                )

        meta = {
            "shard_datalist_dir": self.shard_datalist_dir,
            "shard_datalist_dir_abs": shard_root,
            "num_shards": self.num_shards,
            "fixed_shard_order": self.fixed_shard_order,
            "source_datalist": self.source_datalist,
            "shard_seed": self.shard_seed,
            "total_samples": len(all_lines),
            "duplicate_count": duplicate_count,
            "source_count": source_count,
            "source_match": source_match,
            "shards": [
                {
                    "id": shard["id"],
                    "path": shard["path"],
                    "size": shard["size"],
                }
                for shard in shards
            ],
        }
        return shards, meta

    def _update_run_artifacts_for_shards(self):
        config_path = os.path.join(self.run_dir, "config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            config_data = {}
        config_data["shard_rotation"] = self._shard_meta
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {"run_id": self.run_id, "summary": {}}
        manifest["kind"] = "train_shard_rotation"
        manifest["shard_rotation"] = self._shard_meta
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _write_shard_schedule(self):
        payload = {
            "run_id": self.run_id,
            "num_shards": self.num_shards,
            "fixed_shard_order": self.fixed_shard_order,
            "source_datalist": self.source_datalist,
            "shard_seed": self.shard_seed,
            "shards": self._shard_meta.get("shards", []),
            "epochs": self._shard_schedule,
        }
        with open(self._shard_schedule_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _shard_order_for_epoch(self, abs_ep: int) -> List[int]:
        order = list(range(self.num_shards))
        if self.fixed_shard_order:
            return order
        rng = np.random.RandomState(self.shard_seed + abs_ep)
        return [int(x) for x in rng.permutation(order)]

    def _set_train_datapath_to_shard(self, shard_id: int):
        shard = self._shards[shard_id]
        datapath = self.dataset_module.train_datapath
        assert datapath is not None, "train_datapath is None"
        datapath.filepaths = list(shard["lines"])
        datapath.use_prob = False
        datapath.num_files = None
        datapath.cls_name = ["shapenet"]
        datapath.cls_bias = [0]
        datapath.cls_length = [len(shard["lines"])]
        datapath.cls_weight = [1.0]
        if hasattr(datapath, "perms"):
            delattr(datapath, "perms")
        if hasattr(datapath, "current_bias"):
            delattr(datapath, "current_bias")
        if self.dataset_module.train_dataset_config is not None:
            self.dataset_module.train_dataset_config.datapath = datapath

    def _run_validation_once(self, abs_ep: int):
        self.model.eval()
        val_dl = self.dataset_module.validate_dataloader()
        if val_dl is None:
            return
        self.on_validation_epoch_start()
        if isinstance(val_dl, dict):
            for name, dl in val_dl.items():
                for batch in tqdm(
                    dl,
                    total=len(dl) // dl.batch_size,
                    desc=f"Epoch {abs_ep}, Validate {name}",
                ):
                    self.on_validation_batch_start()
                    self.validation_step(batch)
                    self.on_validation_batch_end()
        else:
            for batch in tqdm(
                val_dl,
                total=len(val_dl) // val_dl.batch_size,
                desc=f"Epoch {abs_ep}, Validate",
            ):
                self.on_validation_batch_start()
                self.validation_step(batch)
                self.on_validation_batch_end()
        self.on_validation_epoch_end()

    def train(self):
        self.ckpt_save_dir = os.path.join(self.run_dir, "checkpoints")
        self.model.set_predict(False)
        assert self.optimizer is not None, "optimizer is None, cannot train"

        for epoch in range(self.epochs):
            abs_ep = self._start_epoch + epoch
            shard_order = self._shard_order_for_epoch(abs_ep)
            self.model.train()
            self.on_train_epoch_start()

            shard_batch_counts: List[int] = []
            for shard_id in shard_order:
                self._set_train_datapath_to_shard(shard_id)
                shard = self._shards[shard_id]
                train_dl = self.dataset_module.train_dataloader()
                assert train_dl is not None
                batch_count = 0
                pbar = tqdm(
                    train_dl,
                    total=len(train_dl) // train_dl.batch_size,
                    desc=f"Epoch {abs_ep}, Shard {shard_id:02d}",
                )
                for batch_idx, batch in enumerate(pbar):
                    self.on_train_batch_start()
                    loss = self.training_step(batch)
                    self.optimizer.zero_grad()
                    self.optimizer.backward(loss)
                    if batch_idx % self._metrics_sample_interval == 0:
                        pbar.set_description(
                            f"Epoch {abs_ep}, Shard {shard_id:02d}, "
                            f"Loss: {_get_item(loss)}"
                        )
                    else:
                        pbar.set_description(f"Epoch {abs_ep}, Shard {shard_id:02d}")
                    self.on_before_optimizer_step(self.optimizer)
                    self.optimizer.step()
                    self.on_train_batch_end()
                    batch_count += 1
                shard_batch_counts.append(batch_count)

            shard_sizes = [self._shards[shard_id]["size"] for shard_id in shard_order]
            total_samples = sum(shard_sizes)
            total_batches = sum(shard_batch_counts)
            schedule_entry = {
                "epoch": self._current_epoch,
                "abs_epoch": abs_ep,
                "shard_order": shard_order,
                "shard_sizes": shard_sizes,
                "shard_batch_counts": shard_batch_counts,
                "total_samples": total_samples,
                "total_batches": total_batches,
            }
            self._shard_schedule.append(schedule_entry)
            self._write_shard_schedule()
            self._epoch_extra_summary = {
                "shard_rotation_num_shards": self.num_shards,
                "shard_rotation_total_samples": total_samples,
                "shard_rotation_total_batches": total_batches,
                "shard_rotation_order": shard_order,
            }
            self.on_train_epoch_end()

            self._run_validation_once(abs_ep)

            ckpt_path = os.path.join(
                self.ckpt_save_dir, f"{self.ckpt_save_name}_{abs_ep}.pkl"
            )
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(ckpt_path)
            self._save_strict_resume_state(ckpt_path, abs_ep)


# ===========================================================================
# PDLTSLightPredictSystem: 正式预测路径 
# ===========================================================================
class PDLTSLightPredictSystem(DummySystem):
    """PDLTS Light 的预测 system. 落实 推理落盘规则 的硬要求:

    1. 日志契约:
         outputs/predictions/<stage>/<run_id>/
             command.sh / config.yaml / env.txt / git_commit.txt / manifest.json
             logs/predict.log
             pred/shapenet/<cls>/<id>/denoised.npy  (由 writer 写)

    2. 三层 seed_k 策略 (基于 coverage sweep 2026-05-08 更新):
         L1: 默认 seed_k=12 (200 mock_test 样本 100% zero-miss)
         L2: L1 漏点则自动重试 seed_k=16 (留作保险, sweep 下不触发)
         L3: L2 仍漏点接受回填, 记入 degraded_samples

    3. 绿/黄/红灯汇总:
         绿: n_missing_total == 0
         黄: max(missing_ratio) < threshold_yellow (默认 1e-4)
         红: 超过阈值, manifest.summary.verdict = "red", 不可直接提交

    关键字段 (构造参数):
        run_id: run 目录名. 未指定则 YYYYMMDD_HHMMSS_<run_tag>_predict;
            新 predict run_tag 建议采用 a_final_<stage>_<scope>.
        run_tag: 默认 "a_final"; 由 system yaml 覆盖.
        run_output_root: 默认 "../outputs/predictions".
        seed_k_l1 / seed_k_l2: 默认 12 / 16 (coverage sweep 后确定).
        threshold_yellow: 默认 1e-4 (0.01% 漏点以下算黄灯).

    与 PDLTSLightSystem 不共享 run 目录 (predictions 和 runs 分离).
    writer 仍然是 PDLTSLightWriter, 但 save_dir 被 system 锁定为 run_dir/pred.
    """

    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
        run_id: Optional[str] = None,
        run_tag: str = "a_final",
        run_output_root: str = "../outputs/predictions",
        seed_k_l1: int = 12,
        seed_k_l2: int = 16,
        threshold_yellow: float = 1e-4,
        load_ckpt: Optional[str] = None,
        postselect_mode: str = "off",
        candidate_generator_mode: str = "off",
        selector_mode: str = "identity",
        refine_mode: str = "off",
        candidate_ratio: float = 1.5,
        candidate_knn_k: int = 8,
        candidate_surface_reject: bool = False,
        candidate_reject_k: int = 8,
        candidate_reject_spacing_mult: float = 2.5,
        candidate_reject_abs_threshold: float = 0.0,
        replace_ratio: float = 0.02,
        replace_drop_spacing_k: int = 8,
        fps_exact_limit: int = 4096,
        postselect_seed: int = 0,
        CONFIG_VERSION: Optional[str] = None,
    ):
        # 不建 optimizer (predict 不训).
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.optimizer = None
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get("epochs", 1)
        self._validation_loss = defaultdict(list)
        model_config = getattr(model, "model_config", {}) if model is not None else {}
        network = getattr(model, "network", None) if model is not None else None
        self.predict_return_pull_sidecars = bool(model_config.get(
            "predict_return_pull_sidecars",
            getattr(network, "pull_head_mode", "off") != "off",
        ))
        self.predict_pull_head_mode = str(getattr(network, "pull_head_mode", "off"))
        self.predict_pull_head_feat_source = str(
            getattr(network, "pull_head_feat_source", model_config.get("pull_head_feat_source", "predict_z"))
        )

        # run_id / run_dir (predictions 和 runs 分离)
        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{run_tag}_predict"
        self.run_id = run_id
        self.run_dir = _versioned_output_dir(
            run_output_root, run_id, run_tag, CONFIG_VERSION
        )
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "pred"), exist_ok=True)

        # 日志文件
        self._log_predict_path = os.path.join(self.run_dir, "logs", "predict.log")
        open(self._log_predict_path, "w").close()

        # 元数据
        _write_command_sh(self.run_dir)
        _write_env_txt(self.run_dir)
        _write_git_commit(self.run_dir)
        _write_config_snapshot(self.run_dir, {
            "run_id": run_id,
            "run_tag": run_tag,
            "run_output_root": run_output_root,
            "seed_k_l1": seed_k_l1,
            "seed_k_l2": seed_k_l2,
            "threshold_yellow": threshold_yellow,
            "load_ckpt": load_ckpt,        # 追溯到训练 run (推理落盘规则 硬要求)
            "postselect_mode": postselect_mode,
            "candidate_generator_mode": candidate_generator_mode,
            "selector_mode": selector_mode,
            "refine_mode": refine_mode,
            "candidate_ratio": candidate_ratio,
            "candidate_knn_k": candidate_knn_k,
            "candidate_surface_reject": candidate_surface_reject,
            "candidate_reject_k": candidate_reject_k,
            "candidate_reject_spacing_mult": candidate_reject_spacing_mult,
            "candidate_reject_abs_threshold": candidate_reject_abs_threshold,
            "replace_ratio": replace_ratio,
            "replace_drop_spacing_k": replace_drop_spacing_k,
            "fps_exact_limit": fps_exact_limit,
            "postselect_seed": postselect_seed,
            "predict_return_pull_sidecars": self.predict_return_pull_sidecars,
            "pull_head_mode": self.predict_pull_head_mode,
            "pull_head_feat_source": self.predict_pull_head_feat_source,
            "CONFIG_VERSION": CONFIG_VERSION,
        })
        leakage_flags = {
            "predict_reads_clean": False,
            "predict_reads_mesh": False,
            "predict_reads_norm": False,
            "predict_reads_oracle": False,
            "predict_reads_target_cache": False,
            "predict_reads_frozen_prediction": False,
            "predict_reads_candidate_pool": bool(
                str(postselect_mode) != "off" or str(candidate_generator_mode) != "off"
            ),
            "predict_uses_postselect": bool(str(postselect_mode) != "off"),
            "predict_uses_candidate_generation": bool(str(candidate_generator_mode) != "off"),
        }
        self.manifest_path = _write_manifest(
            self.run_dir,
            extra={
                "kind": "predict",
                "load_ckpt": load_ckpt,
                "postselect_mode": postselect_mode,
                "candidate_generator_mode": candidate_generator_mode,
                "selector_mode": selector_mode,
                "refine_mode": refine_mode,
                "predict_return_pull_sidecars": self.predict_return_pull_sidecars,
                "pull_head_mode": self.predict_pull_head_mode,
                "pull_head_feat_source": self.predict_pull_head_feat_source,
                "CONFIG_VERSION": CONFIG_VERSION,
                "leakage_flags": leakage_flags,
            },
        )
        _write_notes_md(self.run_dir)

        # 持久化 load_ckpt 到 system 实例, 方便 _finalize_manifest 补写.
        self.load_ckpt = load_ckpt

        # 把 writer 的 save_dir 锁到 run_dir/pred (不允许 task yaml 覆盖).
        # writer 即使是 PDLTSLightWriter 也可能拿到不一致的 save_dir; 这里强制覆盖.
        self.writer = writer
        if isinstance(self.writer, PDLTSLightWriter):
            forced_save_dir = os.path.join(self.run_dir, "pred")
            self.writer.save_dir = forced_save_dir

        # 三层策略参数
        self.seed_k_l1 = seed_k_l1
        self.seed_k_l2 = seed_k_l2
        self.threshold_yellow = threshold_yellow
        self.postselect_mode = str(postselect_mode)
        self.candidate_generator_mode = str(candidate_generator_mode)
        self.selector_mode = str(selector_mode)
        self.refine_mode = str(refine_mode)
        self.candidate_ratio = float(candidate_ratio)
        self.candidate_knn_k = int(candidate_knn_k)
        self.candidate_surface_reject = bool(candidate_surface_reject)
        self.candidate_reject_k = int(candidate_reject_k)
        self.candidate_reject_spacing_mult = float(candidate_reject_spacing_mult)
        self.candidate_reject_abs_threshold = float(candidate_reject_abs_threshold)
        self.replace_ratio = float(replace_ratio)
        self.replace_drop_spacing_k = int(replace_drop_spacing_k)
        self.fps_exact_limit = int(fps_exact_limit)
        self.postselect_seed = int(postselect_seed)

        # 每样本 summary
        self._sample_records: List[Dict] = []

    def _log_line(self, line: str):
        with open(self._log_predict_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _apply_postselect(self, denoised_np: np.ndarray, pc_noisy_np: np.ndarray):
        """Post-stitch full-cloud selection hook.

        The default off mode is an exact identity path and exists only in
        predict system scope, after denoise_full_cloud() has produced base_N.
        """
        from ..model.pdlts_light.postselect import PostSelectConfig, apply_postselect

        cfg = PostSelectConfig(
            postselect_mode=self.postselect_mode,
            candidate_generator_mode=self.candidate_generator_mode,
            selector_mode=self.selector_mode,
            refine_mode=getattr(self, "refine_mode", "off"),
            candidate_ratio=self.candidate_ratio,
            candidate_knn_k=getattr(self, "candidate_knn_k", 8),
            candidate_surface_reject=getattr(self, "candidate_surface_reject", False),
            candidate_reject_k=getattr(self, "candidate_reject_k", 8),
            candidate_reject_spacing_mult=getattr(self, "candidate_reject_spacing_mult", 2.5),
            candidate_reject_abs_threshold=getattr(self, "candidate_reject_abs_threshold", 0.0),
            replace_ratio=getattr(self, "replace_ratio", 0.02),
            replace_drop_spacing_k=getattr(self, "replace_drop_spacing_k", 8),
            fps_exact_limit=getattr(self, "fps_exact_limit", 4096),
            seed=self.postselect_seed,
        )
        return apply_postselect(denoised_np, pc_noisy_np, cfg)

    def _run_single_sample(self, pc_noisy_np: np.ndarray) -> Dict:
        """对单个 noisy 点云跑三层策略, 返回:
           {denoised_np, n_missing, missing_ratio, seed_k_used, level}
        """
        from ..model.pdlts_light.denoise import denoise_full_cloud

        # 从 model_config 读取 patch_size / seed_k_alpha
        mc = self.model.model_config
        patch_size = int(mc.get("patch_size", 1024))
        seed_k_alpha = int(mc.get("predict_seed_k_alpha", 5))
        return_pull_sidecars = self.predict_return_pull_sidecars

        def _run_denoise(seed_k: int):
            denoise_out = denoise_full_cloud(
                self.model.network, pc_noisy_np,
                patch_size=patch_size, seed_k=seed_k,
                seed_k_alpha=seed_k_alpha, return_coverage=True,
                return_sidecars=return_pull_sidecars,
            )
            if return_pull_sidecars:
                denoised, coverage, sidecars = denoise_out
            else:
                denoised, coverage = denoise_out
                sidecars = None
            return denoised, coverage, sidecars

        def _finish_record(denoised_np, info, sidecars, seed_k_used: int, level: str):
            denoised_np, post_info = self._apply_postselect(denoised_np, pc_noisy_np)
            if sidecars is not None and post_info.get("enabled", False):
                raise RuntimeError(
                    "pull sidecars are only gate-eligible when postselect is off; "
                    "postselect changes the full-cloud slots after sidecar stitching"
                )
            rec = {
                "denoised_np": denoised_np,
                "n_missing": int(info["n_missing"]),
                "missing_ratio": float(info["missing_ratio"]),
                "seed_k_used": int(seed_k_used),
                "level": level,
                "postselect_info": post_info,
            }
            if sidecars is not None:
                rec.update(sidecars)
                rec["pull_input_audit"] = getattr(self.model.network, "_pull_input_audit", {})
            return rec

        # L1
        denoised_np, info, sidecars = _run_denoise(self.seed_k_l1)
        if info["n_missing"] == 0:
            return _finish_record(
                denoised_np, info, sidecars, seed_k_used=self.seed_k_l1, level="L1"
            )

        # L2 重试
        denoised_np, info, sidecars = _run_denoise(self.seed_k_l2)
        level = "L2" if info["n_missing"] == 0 else "L3"
        return _finish_record(
            denoised_np, info, sidecars, seed_k_used=self.seed_k_l2, level=level
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Override DummySystem.predict_step: 不经 model.predict_step, 而是走三层策略."""
        pc_noisy_batch = batch["pc_noisy"]  # (B, N, 3)
        assets = batch["asset"]
        res = []
        for i in range(pc_noisy_batch.shape[0]):
            pc_noisy = pc_noisy_batch[i]
            pc_noisy_np = pc_noisy.numpy().astype(np.float32)
            t0 = time.time()
            rec = self._run_single_sample(pc_noisy_np)
            elapsed = time.time() - t0

            # 记录
            asset_path = assets[i].path if i < len(assets) and assets[i].path else "unknown"
            sample_record = {
                "sample_path": asset_path,
                "n_points": int(pc_noisy_np.shape[0]),
                "n_missing": rec["n_missing"],
                "missing_ratio": rec["missing_ratio"],
                "seed_k_used": rec["seed_k_used"],
                "level": rec["level"],
                "elapsed_sec": round(elapsed, 3),
            }
            post_info = rec.get("postselect_info", {})
            if post_info:
                sample_record.update({
                    "postselect_enabled": bool(post_info.get("enabled", False)),
                    "postselect_mode": post_info.get("postselect_mode", "off"),
                    "postselect_selector_mode": post_info.get("selector_mode", "off"),
                    "postselect_refine_mode": post_info.get("refine_mode", "off"),
                    "postselect_candidate_impl": post_info.get("candidate_generator_impl", "off"),
                    "postselect_num_candidates": int(post_info.get("num_candidates", 0)),
                    "postselect_num_extra": int(post_info.get("num_extra", 0)),
                    "postselect_extra_reject_rate": float(post_info.get("extra_reject_rate", 0.0)),
                    "postselect_keep_n": int(post_info.get("keep_n", 0)),
                    "postselect_replace_count_actual": int(
                        post_info.get("replace_count_actual", 0)),
                    "postselect_coverage_reference": post_info.get(
                        "coverage_reference", "base_stitching"),
                    "postselect_realized_coverage_recomputed": bool(
                        post_info.get("realized_coverage_recomputed", False)),
                })
            if "confidence_w" in rec:
                confidence_w = np.asarray(rec["confidence_w"], dtype=np.float64)
                sample_record.update({
                    "sidecar_confidence_w_present": True,
                    "sidecar_confidence_w_finite": bool(np.isfinite(confidence_w).all()),
                    "sidecar_confidence_w_min": float(confidence_w.min()),
                    "sidecar_confidence_w_max": float(confidence_w.max()),
                    "sidecar_confidence_w_mean": float(confidence_w.mean()),
                    "sidecar_confidence_w_var": float(confidence_w.var()),
                })
            else:
                sample_record["sidecar_confidence_w_present"] = False
            if "pull_norm" in rec:
                pull_norm = np.asarray(rec["pull_norm"], dtype=np.float64)
                sample_record.update({
                    "sidecar_pull_norm_present": True,
                    "sidecar_pull_norm_finite": bool(np.isfinite(pull_norm).all()),
                    "sidecar_pull_norm_mean": float(pull_norm.mean()),
                    "sidecar_pull_norm_p95": float(np.percentile(pull_norm, 95)),
                    "sidecar_pull_norm_max": float(pull_norm.max()),
                })
            else:
                sample_record["sidecar_pull_norm_present"] = False
            self._sample_records.append(sample_record)
            self._log_line(json.dumps(sample_record, ensure_ascii=False))

            # shape 硬断言
            assert rec["denoised_np"].shape == pc_noisy_np.shape, (
                f"predict_step shape mismatch: in={pc_noisy_np.shape}, "
                f"out={rec['denoised_np'].shape}"
            )
            out_rec = {
                "pc_denoised": rec["denoised_np"],
                "coverage_info": {
                    "n_missing": rec["n_missing"],
                    "missing_ratio": rec["missing_ratio"],
                    "seed_k_used": rec["seed_k_used"],
                    "level": rec["level"],
                },
            }
            for sidecar_key in ("confidence_w", "pull_norm", "pull_vector"):
                if sidecar_key in rec:
                    out_rec[sidecar_key] = rec[sidecar_key]
            if "pull_input_audit" in rec:
                out_rec["pull_input_audit"] = rec["pull_input_audit"]
            res.append(out_rec)
        return res

    def _finalize_manifest(self):
        """predict 全部样本跑完后汇总 manifest."""
        n_total = len(self._sample_records)
        n_missing_total = sum(r["n_missing"] for r in self._sample_records)
        n_missing_samples = sum(1 for r in self._sample_records if r["n_missing"] > 0)
        max_missing_ratio = max(
            (r["missing_ratio"] for r in self._sample_records), default=0.0
        )
        levels_count = defaultdict(int)
        for r in self._sample_records:
            levels_count[r["level"]] += 1
        degraded_samples = [
            r["sample_path"] for r in self._sample_records if r["level"] == "L3"
        ]
        postselect_enabled_samples = sum(
            1 for r in self._sample_records if r.get("postselect_enabled", False)
        )
        sidecar_conf_samples = [
            r for r in self._sample_records if r.get("sidecar_confidence_w_present", False)
        ]
        sidecar_pull_samples = [
            r for r in self._sample_records if r.get("sidecar_pull_norm_present", False)
        ]

        # Verdict
        if n_missing_total == 0:
            verdict = "green"
        elif max_missing_ratio < self.threshold_yellow:
            verdict = "yellow"
        else:
            verdict = "red"

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"run_id": self.run_id, "summary": {}}
        data["summary"] = {
            "n_samples": n_total,
            "n_missing_total": int(n_missing_total),
            "n_missing_samples": int(n_missing_samples),
            "max_missing_ratio": float(max_missing_ratio),
            "levels_count": dict(levels_count),
            "degraded_samples": degraded_samples,
            "verdict": verdict,
            "threshold_yellow": self.threshold_yellow,
            "postselect_enabled_samples": int(postselect_enabled_samples),
            "predict_return_pull_sidecars": bool(self.predict_return_pull_sidecars),
            "sidecar_confidence_w_samples": int(len(sidecar_conf_samples)),
            "sidecar_pull_norm_samples": int(len(sidecar_pull_samples)),
        }
        if sidecar_conf_samples:
            data["summary"].update({
                "sidecar_confidence_w_all_finite": bool(
                    all(r.get("sidecar_confidence_w_finite", False) for r in sidecar_conf_samples)
                ),
                "sidecar_confidence_w_min": float(
                    min(r["sidecar_confidence_w_min"] for r in sidecar_conf_samples)
                ),
                "sidecar_confidence_w_max": float(
                    max(r["sidecar_confidence_w_max"] for r in sidecar_conf_samples)
                ),
                "sidecar_confidence_w_mean_mean": float(
                    np.mean([r["sidecar_confidence_w_mean"] for r in sidecar_conf_samples])
                ),
                "sidecar_confidence_w_var_mean": float(
                    np.mean([r["sidecar_confidence_w_var"] for r in sidecar_conf_samples])
                ),
            })
        if sidecar_pull_samples:
            data["summary"].update({
                "sidecar_pull_norm_all_finite": bool(
                    all(r.get("sidecar_pull_norm_finite", False) for r in sidecar_pull_samples)
                ),
                "sidecar_pull_norm_mean_mean": float(
                    np.mean([r["sidecar_pull_norm_mean"] for r in sidecar_pull_samples])
                ),
                "sidecar_pull_norm_p95_mean": float(
                    np.mean([r["sidecar_pull_norm_p95"] for r in sidecar_pull_samples])
                ),
                "sidecar_pull_norm_max": float(
                    max(r["sidecar_pull_norm_max"] for r in sidecar_pull_samples)
                ),
            })
        if postselect_enabled_samples > 0:
            data["summary"].update({
                "postselect_coverage_reference": "preselect_base_stitching",
                "postselect_realized_coverage_recomputed": False,
                "postselect_coverage_note": (
                    "n_missing/missing_ratio describe the pre-select base "
                    "stitching coverage, not the realized postselect output."
                ),
            })
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._log_line(
            f"[summary] verdict={verdict}  n_samples={n_total}  "
            f"n_missing_total={n_missing_total}  "
            f"max_missing_ratio={max_missing_ratio:.2e}"
        )

    def predict(self):
        """复用 DummySystem.predict 的主循环, 但结束后写 manifest summary."""
        super().predict()
        self._finalize_manifest()
