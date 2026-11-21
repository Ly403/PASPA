# BEAT2 best-checkpoint artifacts / 最佳权重实验快照

This directory preserves the original artifacts associated with the released
BEAT2 `PID 2` and `PID all` checkpoints. Both checkpoints were produced with
the same training recipe.

本目录保存随 BEAT2 `PID 2` 与 `PID all` 最佳权重归档的原始实验文件；两个权重
来自同一套训练配方。

- `best_checkpoint_config.yaml`: the saved, fully resolved Hydra configuration.
  It intentionally retains historical machine-specific paths and should be used
  for provenance and parameter inspection, not as a portable configuration.
- `original_best_recipe.sh`: the original shell snapshot. Despite its historical
  `train_...` filename, it calls `scripts/eval_pth_dir.py` to load and select
  checkpoints; it is not the release training entrypoint.
- `../../train_scripts/train_beat2.sh`: the portable release launcher for new
  training runs. It exposes dataset and asset paths through environment variables.

- `best_checkpoint_config.yaml`：训练保存的完整 Hydra 配置快照。文件有意保留
  当时机器上的绝对路径，仅用于追溯和核对参数，不应作为可移植配置直接运行。
- `original_best_recipe.sh`：当时保存的 shell 快照。虽然原文件名以 `train_`
  开头，但脚本实际调用 `scripts/eval_pth_dir.py` 加载和筛选 checkpoint，并不是
  发行版训练入口。
- `../../train_scripts/train_beat2.sh`：新训练应使用的可移植发行入口；数据和资产
  路径均通过环境变量传入。
