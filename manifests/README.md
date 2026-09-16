# 发布文件清单

[release_file_manifest.json](release_file_manifest.json) 记录公开目录内每个文件的路径、字节数和 SHA-256（哈希校验值），不包含清单自身。它用于检查当前目录是否完整，与 [B 榜历史比赛提交清单](../b_board/results/README.md)用途不同。

从仓库根目录执行：

```bash
python -B validation/prepare_release.py --version pdlts_source_20260916 --verify-only
```

有文件变动时先执行相关验证与静态检查，再用 `--manifest-only` 刷新清单。清单中的 `validated_source` 表示源码快照通过所列检查，不代表重新完成训练或线上评测。
