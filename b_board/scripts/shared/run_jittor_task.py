"""用受控的 Jittor 算子编译并发执行一个公开 task（任务）。"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    starter_root = Path.cwd().resolve()
    run_file = starter_root / "run.py"
    if not run_file.is_file():
        raise SystemExit(
            f"请从 b_board/starter_code 运行此入口；未找到 {run_file}"
        )

    try:
        compiler_threads = int(os.environ.get("JITTOR_COMPILER_THREADS", "1"))
    except ValueError as exc:
        raise SystemExit("JITTOR_COMPILER_THREADS 必须是非负整数") from exc
    if compiler_threads < 0:
        raise SystemExit("JITTOR_COMPILER_THREADS 必须是非负整数")
    if len(sys.argv) < 2:
        raise SystemExit("用法：python run_jittor_task.py --task <task>")

    import jittor as jt

    # 某些 Jittor/CUDA 环境在连续启动两次大模型时会在并行编译阶段崩溃。
    # 默认单路编译只影响首次算子编译，不改变模型结构、参数或推理逻辑。
    jt.flags.use_parallel_op_compiler = compiler_threads

    starter_text = str(starter_root)
    if starter_text not in sys.path:
        sys.path.insert(0, starter_text)
    sys.argv = [str(run_file), *sys.argv[1:]]
    runpy.run_path(str(run_file), run_name="__main__")


if __name__ == "__main__":
    main()
