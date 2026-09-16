"""PDLTSLightSystem smoke 单测。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_pdlts_light_system

验证内容:
    A. 可训练参数过滤
        A1. is_trainable_param_name 对已知状态变量返回 False
        A2. collect_trainable_params(model) 产出列表不含状态变量
        A3. make_filtered_optimizer 构建的 optimizer 参数数 == 过滤后数目
    B. 日志契约
        B1. PDLTSLightSystem(...) 构造后 run_dir/ 存在, 并有:
            command.sh / env.txt / git_commit.txt / config.yaml / manifest.json
            logs/train.log, logs/epoch_summary.jsonl
            checkpoints/ 空目录
    C. Ckpt save/load
        C1. 训练后 (1 epoch) ckpt 文件存在
        C2. 重新构造 model + load ckpt 后, channel_mask 末尾 24 位仍 = 0
        C3. 所有 ActNorm1d 的 is_inited 仍 = 1 (意味着 inverse 路径不阻塞)
    D. 1 epoch 实跑 (合成 dataloader, 不启真数据)
        D1. 跑 1 epoch, 每 step loss 有限
        D2. epoch_summary.jsonl 有 1 行, 含 loss_sum_mean
        D3. manifest.json 的 summary 字典含 epoch_0

注:
    - 使用合成数据检查 System 层，不通过 run.py 启动完整训练。
    - 不测 validate (validate 路径需要 asset/cls collate, smoke train 时会跑到).
"""

import sys
import os
import json
import glob
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt
from jittor import nn

from src.system.pdlts_light import (
    PDLTSLightSystem,
    is_trainable_param_name,
    collect_trainable_params,
    make_filtered_optimizer,
)
from src.model.parse import get_model


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


# 小号 model 以控制测试时长
SMALL_MODEL_CFG = {
    "__target__": "PDLTSLight",
    "pc_channel": 3,
    "aug_channel": 48,
    "n_injector": 12,
    "cut_channel": 24,
    "nflow_module": 12,
    "num_neighbors": 8,
    "mlgc_hidden": 32,
    "coupling_hidden": 32,
    "log_scale_clamp": 0.1,
}


def _make_model():
    cfg = dict(SMALL_MODEL_CFG)
    return get_model(model_config=cfg, transform_config={})


# ============================================================================
# A. 可训练参数过滤
# ============================================================================

def test_is_trainable_param_name():
    # 应排除的
    assert not is_trainable_param_name("PreConv.conv.1.running_mean")
    assert not is_trainable_param_name("feat_Conv.0.convs.0.1.running_var")
    assert not is_trainable_param_name("flow_assemblies.0.chain.0.is_inited")
    assert not is_trainable_param_name("flow_assemblies.0.chain.1.mask")
    assert not is_trainable_param_name("channel_mask")
    assert not is_trainable_param_name("network.channel_mask")
    # 应保留的 (抽几个真实 name)
    assert is_trainable_param_name("PreConv.conv.0.weight")
    assert is_trainable_param_name("feat_Conv.0.convs.0.0.weight")
    assert is_trainable_param_name("flow_assemblies.0.chain.0.weight")   # ActNorm weight
    assert is_trainable_param_name("flow_assemblies.0.chain.1.net.0.weight")
    print("[PASS] test_is_trainable_param_name")


def test_collect_trainable_params_excludes_buffers():
    model = _make_model()
    all_params = list(model.named_parameters())
    trainable = collect_trainable_params(model)
    # 过滤后 count 严格小于总 count
    assert len(trainable) < len(all_params), (
        f"trainable ({len(trainable)}) should be < all ({len(all_params)})"
    )
    # trainable 里不应出现任何状态变量名 (通过 id 对应回去)
    trainable_ids = {id(p) for p in trainable}
    for name, p in all_params:
        if not is_trainable_param_name(name):
            assert id(p) not in trainable_ids, \
                f"state-var '{name}' leaked into trainable list"
    leaked = [name for name, p in all_params if "channel_mask" in name and id(p) in trainable_ids]
    assert not leaked, f"channel_mask leaked into trainable list: {leaked}"
    print(f"[PASS] test_collect_trainable_params_excludes_buffers  "
          f"trainable={len(trainable)}  all={len(all_params)}")


def test_make_filtered_optimizer_param_count():
    model = _make_model()
    optimizer = make_filtered_optimizer({"__target__": "adam", "lr": 1e-3}, model)
    # Jittor optim 把参数按 param_group 存, 比较总数
    n_in_opt = sum(len(g["params"]) for g in optimizer.param_groups)
    n_trainable = len(collect_trainable_params(model))
    assert n_in_opt == n_trainable, \
        f"optimizer param count {n_in_opt} != trainable {n_trainable}"
    print(f"[PASS] test_make_filtered_optimizer_param_count  n={n_in_opt}")


# ============================================================================
# B. 日志契约
# ============================================================================

def test_run_dir_contract():
    model = _make_model()
    tmp = tempfile.mkdtemp(prefix="system_contract_test_")
    try:
        sys_ = PDLTSLightSystem(
            dataset_module=None,
            model=model,
            loss_config={"chamfer": 1.0, "l2": 0.1},
            optimizer_config={"__target__": "adam", "lr": 1e-3},
            trainer_config={"epochs": 1},
            run_id="run_dir_contract_test",
            run_output_root=tmp,
        )
        rd = sys_.run_dir
        for fname in ("command.sh", "env.txt", "git_commit.txt", "config.yaml",
                      "manifest.json", "notes.md"):
            assert os.path.exists(os.path.join(rd, fname)), f"missing {fname}"
        for fname in ("logs/train.log", "logs/epoch_summary.jsonl", "logs/metrics.csv"):
            assert os.path.exists(os.path.join(rd, fname)), f"missing {fname}"
        assert os.path.isdir(os.path.join(rd, "checkpoints"))
        # config.yaml 能 json load
        with open(os.path.join(rd, "config.yaml"), encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["run_id"] == "run_dir_contract_test"
        assert cfg["loss_config"] == {"chamfer": 1.0, "l2": 0.1}
        # manifest.json 结构
        with open(os.path.join(rd, "manifest.json"), encoding="utf-8") as f:
            mani = json.load(f)
        assert mani["run_id"] == "run_dir_contract_test"
        assert mani["kind"] == "train"
        print(f"[PASS] test_run_dir_contract  dir={os.path.relpath(rd, tmp)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# C. Ckpt save / load 保真 (channel_mask + is_inited)
# ============================================================================

def _synthetic_batch(B=2, P=1, M=32):
    return {
        "pc_noisy": jt.randn(B, P, M, 3),
        "pc_clean": jt.randn(B, P, M, 3),
    }


def _train_one_step(system):
    """模拟一步训练: forward + backward + step. (不走 train() 的 dataloader 路径.)"""
    batch = _synthetic_batch()
    # 触发 ActNorm 首次初始化
    _ = system.training_step(batch)
    # 真正 backward
    batch = _synthetic_batch()
    loss = system.training_step(batch)
    system.optimizer.zero_grad()
    system.optimizer.backward(loss)
    system.optimizer.step()


def test_ckpt_save_load_preserves_fixed_state():
    model = _make_model()
    tmp = tempfile.mkdtemp(prefix="system_contract_ckpt_")
    try:
        sys_ = PDLTSLightSystem(
            dataset_module=None,
            model=model,
            loss_config={"chamfer": 1.0, "l2": 0.1},
            optimizer_config={"__target__": "adam", "lr": 1e-3},
            trainer_config={"epochs": 1},
            run_id="ckpt_test",
            run_output_root=tmp,
        )
        _train_one_step(sys_)
        # 手动 save (模拟 train() 主循环的 ckpt 行为)
        ckpt_dir = os.path.join(sys_.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "pdlts_light_0.pkl")
        model.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        # 重建 model + load
        model2 = _make_model()
        model2.load(ckpt_path)

        # channel_mask 的末尾 24 位应仍 = 0
        mask_np = model2.network.channel_mask.numpy()
        cut = mask_np[..., -24:]
        non_cut = mask_np[..., :-24]
        assert (cut == 0.0).all(), "loaded channel_mask cut region not zero"
        assert (non_cut == 1.0).all(), "loaded channel_mask non-cut region not one"

        # 所有 24 个 ActNorm is_inited 应 = 1 (train_one_step 走过 forward 触发初始化)
        inited_names = []
        for name, p in model2.named_parameters():
            if name.endswith("is_inited"):
                val = float(p.item())
                inited_names.append((name, val))
        assert len(inited_names) == 24, \
            f"expected 24 is_inited (12 FlowAssembly * 2 ActNorm), got {len(inited_names)}"
        for name, val in inited_names:
            assert val >= 0.5, f"{name} not inited after load (value={val})"
        print(f"[PASS] test_ckpt_save_load_preserves_fixed_state  "
              f"channel_mask OK, {len(inited_names)} ActNorm init flags OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_strict_resume_state_restores_epoch_and_optimizer():
    model = _make_model()
    tmp = tempfile.mkdtemp(prefix="pdlts_strict_resume_")
    try:
        sys_ = PDLTSLightSystem(
            dataset_module=None,
            model=model,
            loss_config={"chamfer": 1.0, "l2": 0.1},
            optimizer_config={"__target__": "adam", "lr": 1e-3},
            trainer_config={"epochs": 1},
            run_id="strict_resume_src",
            run_output_root=tmp,
        )
        sys_.on_train_epoch_start()
        _train_one_step(sys_)
        sys_.on_train_epoch_end()
        ckpt_dir = os.path.join(sys_.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "pdlts_light_0.pkl")
        model.save(ckpt_path)
        train_state_path = sys_._save_strict_resume_state(ckpt_path, 0)
        assert os.path.exists(train_state_path), train_state_path

        model2 = _make_model()
        sys2 = PDLTSLightSystem(
            dataset_module=None,
            model=model2,
            loss_config={"chamfer": 1.0, "l2": 0.1},
            optimizer_config={"__target__": "adam", "lr": 1e-3},
            trainer_config={"epochs": 1},
            run_id="strict_resume_dst",
            run_output_root=tmp,
            resume_state=train_state_path,
        )
        assert sys2._start_epoch == 1
        assert sys2._global_step_counter == sys_._global_step_counter
        assert getattr(sys2.optimizer, "n_step", None) == getattr(sys_.optimizer, "n_step", None)
        with open(os.path.join(sys2.run_dir, "config.yaml"), encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["resume_mode"] == "strict"
        assert cfg["resume_state"] == train_state_path
        print("[PASS] test_strict_resume_state_restores_epoch_and_optimizer")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# D. 1 epoch 等效训练 (合成数据, 走 training_step + on_*_epoch hooks)
# ============================================================================

def test_pseudo_one_epoch_logs():
    """不启 DataModule, 手动触发 epoch hooks + 几 step 训练, 验证日志落盘."""
    model = _make_model()
    tmp = tempfile.mkdtemp(prefix="system_contract_epoch_")
    try:
        sys_ = PDLTSLightSystem(
            dataset_module=None,
            model=model,
            loss_config={"chamfer": 1.0, "l2": 0.1},
            optimizer_config={"__target__": "adam", "lr": 1e-3},
            trainer_config={"epochs": 1},
            run_id="pseudo_epoch",
            run_output_root=tmp,
        )
        # 模拟主循环 1 个 epoch + 3 step
        sys_.on_train_epoch_start()
        for _ in range(3):
            batch = _synthetic_batch()
            loss = sys_.training_step(batch)
            val = float(loss.item()) if isinstance(loss, jt.Var) else float(loss)
            assert np.isfinite(val), f"loss not finite: {val}"
            sys_.optimizer.zero_grad()
            sys_.optimizer.backward(loss)
            sys_.optimizer.step()
        sys_.on_train_epoch_end()

        # epoch_summary.jsonl 应有 1 行
        with open(os.path.join(sys_.run_dir, "logs", "epoch_summary.jsonl"),
                  encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 1, f"expected 1 epoch summary line, got {len(lines)}"
        rec = json.loads(lines[0])
        assert rec["epoch"] == 0
        assert "loss_sum_mean" in rec
        assert np.isfinite(rec["loss_sum_mean"])

        # manifest.json 更新
        with open(os.path.join(sys_.run_dir, "manifest.json"),
                  encoding="utf-8") as f:
            mani = json.load(f)
        assert "summary" in mani and "epoch_0" in mani["summary"]
        assert "last_epoch" in mani["summary"]
        assert mani["summary"]["last_epoch"] == 0
        with open(os.path.join(sys_.run_dir, "logs", "train.log"),
                  encoding="utf-8") as f:
            train_log = f.read()
        assert "step 0 loss_sum=" in train_log
        assert "step 2 loss_sum=" in train_log
        with open(os.path.join(sys_.run_dir, "logs", "metrics.csv"),
                  encoding="utf-8") as f:
            metrics_lines = f.read().strip().splitlines()
        assert metrics_lines[0] == "epoch,loss_sum_mean,loss_sum_count"
        assert len(metrics_lines) == 2
        print(f"[PASS] test_pseudo_one_epoch_logs  "
              f"loss_sum_mean={rec['loss_sum_mean']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("system smoke test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    # A. 参数过滤规则
    test_is_trainable_param_name()
    test_collect_trainable_params_excludes_buffers()
    test_make_filtered_optimizer_param_count()
    # B.
    test_run_dir_contract()
    # C.
    test_ckpt_save_load_preserves_fixed_state()
    test_strict_resume_state_restores_epoch_and_optimizer()
    # D.
    test_pseudo_one_epoch_logs()
    print("=" * 60)
    print("ALL PDLTS LIGHT SYSTEM SMOKE TESTS PASSED")
    print("=" * 60)
