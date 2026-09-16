# 发布验证脚本

以下命令从仓库根目录执行。静态检查需要 PyYAML 和 Bash；评测检查使用 NumPy、SciPy，
安装 `point-cloud-utils` 时自动使用 pcu-BVH（点到面距离后端），未安装时使用 NumPy
三角面实现。模型检查还需要 Jittor 及项目依赖。

```bash
python -B validation/evaluate_release_contract.py --output validation/reports/release_evaluator_pcu_validation.json
python -B validation/check_release_entrypoints.py --output validation/reports/release_entrypoint_validation.json
python -B validation/check_release_models.py --output validation/reports/release_model_validation.json
python -B validation/audit_release.py --output validation/reports/release_static_validation.json
```

NumPy 后端的既有记录来自未安装 `point-cloud-utils` 的 Windows 环境，输出名为
`release_evaluator_validation.json`。重新执行时应检查报告中的 `p2s_backend`（评测后端），
两个报告分别对应 NumPy 三角面实现和 pcu-BVH 后端。

Windows 下为入口检查和静态检查传入 `--bash <Git-Bash可执行文件>`。
模型检查使用两个独立 Python 进程加载 A/B 代码，阻止 DCD 模块导入，并检查默认损失、
有限且非零的梯度和禁用分支的错误提示，不启动长训。

依赖解析命令：

```bash
python -m pip install --dry-run --ignore-installed --no-cache-dir --report ../pip_resolution.json -r requirements.txt
```

解析记录见 [环境验证报告](reports/release_environment_validation.json)。
安装清单变动后应重新解析，并更新报告中的来源哈希；该步骤没有创建全新 conda 环境。

检查通过后按 [文件清单与复核](../docs/release_preparation.md) 更新目录清单。
`release_inventory.py` 统一定义排除目录及五个不随包分发的 DCD 对照文件，
`prepare_release.py --manifest-only` 核对报告并更新文件清单，不生成压缩包。
`--verify-only` 检查当前目录是否与清单一致。
