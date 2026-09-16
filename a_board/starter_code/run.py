import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List

import argparse
import json
import numpy as np
import os
import random

from src.data.asset import Asset, Exporter
from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

RESOLVED_CONFIG_PATHS = {}


def resolve_config_path(kind: str, path: str) -> str:
    """解析配置路径；兼容旧的扁平 configs/<kind>/<name>.yaml 写法。"""
    if path.endswith('.yaml'):
        requested = path
    else:
        requested = path + '.yaml'
    if os.path.isfile(requested):
        return requested

    parts = requested.replace("\\", "/").split("/")
    try:
        idx = parts.index(kind)
    except ValueError:
        idx = -1
    if idx > 0 and parts[idx - 1] == "configs":
        search_root = os.path.join(*parts[:idx + 1])
    else:
        search_root = os.path.join("configs", kind)

    basename = os.path.basename(requested)
    matches = []
    if os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            if basename in files:
                matches.append(os.path.join(root, basename))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            f"ambiguous {kind} config {requested!r}; matches={matches}"
        )
    raise FileNotFoundError(
        f"cannot find {kind} config {requested!r}; searched {search_root}"
    )


def load(task: str, path: str) -> Dict:
    resolved = resolve_config_path(task, path)
    RESOLVED_CONFIG_PATHS[task] = resolved
    if resolved != (path if path.endswith('.yaml') else path + '.yaml'):
        print(f"\033[92mload {task} config: {path} -> {resolved}\033[0m")
    else:
        print(f"\033[92mload {task} config: {resolved}\033[0m")
    return OmegaConf.to_container(OmegaConf.load(resolved)) # type: ignore

def debug_fn(data: PCDatasetModule):
    train_dataloader = data.train_dataloader()
    assert train_dataloader is not None, "train_dataloader is None, cannot debug"
    for batch in tqdm(train_dataloader):
        batch: List[Asset]
        # for asset in batch:
        #     Exporter.export_obj(asset.sampled_vertices, "debug.obj")
        #     Exporter.export_obj(asset.sampled_vertices_noisy, "debug_noisy.obj")
        #     exit()

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=123)
    args = parser.parse_args()
    
    # seed all
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    task = load('task', args.task)
    mode = task['mode']
    assert mode in ['train', 'predict', 'debug', 'validate']
    components = task['components']
    
    # get train/validate/predict data
    data_config = load('data', os.path.join('configs/data', components['data']))
    
    # get train dataset
    _train_dataset_config = data_config.get('train_dataset', None)
    if _train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(**_train_dataset_config)
    else:
        train_dataset_config = None
    
    # get validate dataset
    _validate_dataset_config = data_config.get('validate_dataset', None)
    if _validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(**_validate_dataset_config).split_by_cls()
    else:
        validate_dataset_config = None
        
    # get predict dataset
    _predict_dataset_config = data_config.get('predict_dataset', None)
    if _predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(**_predict_dataset_config).split_by_cls()
    else:
        predict_dataset_config = None
    
    # get transform
    transform_config = load('transform', os.path.join('configs/transform', components['transform']))

    # get model
    model_config = components.get('model', None)
    if model_config is None:
        model = None
    else:
        model_config = load('model', os.path.join('configs/model', model_config))
        model = get_model(model_config=model_config, transform_config=transform_config)
    
    train_transform = (Transform.parse(**transform_config.get('train_transform', {}))) if model is None else model.get_train_transform()
    validate_transform = (Transform.parse(**transform_config.get('validate_transform', {}))) if model is None else model.get_validate_transform()
    predict_transform = (Transform.parse(**transform_config.get('predict_transform', {}))) if model is None else model.get_predict_transform()
    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=predict_dataset_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=task.get('debug', False),
    )
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    trainer_config = task.get('trainer', None)
    
    # load ckpt
    load_ckpt = task.get('load_ckpt', None)
    resume_state = task.get('resume_state', None)

    if load_ckpt is not None and model is not None:
        model.load(load_ckpt)

    # get writer
    writer_config = task.get('writer', None)

    # get system
    system_config = components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join('configs/system', system_config))
        # Task-level system overrides are the canonical place for run naming.
        # This avoids creating one-off system yaml files just to change run_tag,
        # and prevents stale predict system configs from leaking wrong output names.
        system_overrides = task.get('system_overrides', {})
        assert isinstance(system_overrides, dict), "task.system_overrides must be a mapping"
        for k, v in system_overrides.items():
            system_config[k] = v
        for k in ("run_tag", "run_id", "run_output_root"):
            if k in task:
                system_config[k] = task[k]
        # 把 load_ckpt 带给 system, 方便 PredictSystem 写进 manifest 追溯
        # (非必需字段, 每个 system 可自行决定是否接收; DummySystem 会忽略).
        extra_kwargs = {}
        if load_ckpt is not None:
            extra_kwargs['load_ckpt'] = load_ckpt
        if resume_state is not None:
            extra_kwargs['resume_state'] = resume_state
        system = get_system(
            dataset_module=dataset_module,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            trainer_config=trainer_config,
            writer=get_writer(**writer_config) if writer_config is not None else None,
            **system_config,
            **extra_kwargs,
        )

        # ---- run-artifact guard: augment system-side config.yaml ----
        # System.__init__ already wrote a config.yaml capturing
        # loss/optimizer/trainer/run_tag/run_id. We merge in seed +
        # task_path + components so the snapshot can reverse-look-up the
        # full execution chain (data/transform/system/model + seed)
        # without re-reading the task yaml.
        run_dir = getattr(system, "run_dir", None)
        if run_dir is not None:
            cfg_path = os.path.join(run_dir, "config.yaml")
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_payload = json.load(f)
            except (FileNotFoundError, ValueError):
                cfg_payload = {}
            cfg_payload.setdefault("seed", args.seed)
            cfg_payload.setdefault(
                "task_path",
                os.path.abspath(RESOLVED_CONFIG_PATHS.get("task", args.task)),
            )
            cfg_payload.setdefault("components", components)
            cfg_payload.setdefault("mode", mode)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_payload, f, indent=2, default=str, ensure_ascii=False)
    else:
        system = None
    
    if mode == 'debug':
        debug_fn(data=dataset_module)
    elif mode == 'train':
        assert system is not None, "system is None, cannot train"
        system.train()
    elif mode == 'predict':
        assert system is not None, "system is None, cannot predict"
        system.predict()
    else:
        assert 0
